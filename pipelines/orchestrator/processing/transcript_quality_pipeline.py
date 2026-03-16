import logging
from pathlib import Path
from typing import Any, Dict

from ai.pipelines.orchestrator.processing.nvidia_clients import (
    NemoCuratorClient,
    NemoEvaluatorClient,
)
from ai.pipelines.orchestrator.processing.voice_transcriber import VoiceTranscriber
from ai.utils.transcript_corrector import TranscriptCorrector

logger = logging.getLogger(__name__)


class TranscriptQualityPipeline:
    """
    Multi-pass transcript quality pipeline (PIX-30).
    Pass 1: ASR (Whisper)
    Pass 2: Correction (NeMo Curator)
    Pass 3: Therapeutic Alignment (NeMo Evaluator)
    """

    def __init__(self, model_name: str = "base"):
        self.transcriber = VoiceTranscriber(
            model_name=model_name, use_faster_whisper=True
        )
        self.curator = NemoCuratorClient()
        self.evaluator = NemoEvaluatorClient()
        self.corrector = TranscriptCorrector()

    def process_audio(self, audio_path: Path) -> Dict[str, Any]:
        """
        Runs the full 3-pass quality pipeline on an audio file.
        """
        logger.info(f"Starting Multi-Pass Quality Pipeline for {audio_path}")

        # Pass 1: Initial Transcription
        transcription_result = self.transcriber.transcribe_audio(str(audio_path))
        if not transcription_result.success:
            return {
                "success": False,
                "error": f"Pass 1 failed: {transcription_result.error_message}",
            }

        initial_text = transcription_result.full_text

        # Pass 2: Correction & Sanitization (Mental Health focus)
        # Using detectable crisis narratives check as a proxy for 'advanced filtering'
        # In a real implementation, we'd have a specific correction NIM
        try:
            # Proxy for correction: using detection to flag issues
            self.curator.detect_crisis_narratives(initial_text)

            # For correction, we'd ideally call an LLM NIM.
            # Since we have Ollama in the docker-compose (even if currently down),
            # or could use external Gemini if configured.
            corrected_text = self._correct_text(initial_text)
        except Exception as e:
            logger.warning(f"Pass 2 remediation failed: {e}. Using raw text.")
            corrected_text = initial_text

        # Pass 3: Therapeutic Alignment
        alignment_report = self._check_therapeutic_alignment(corrected_text)

        return {
            "success": True,
            "original_text": initial_text,
            "corrected_text": corrected_text,
            "alignment": alignment_report,
            "metadata": {
                "whisper_conf": transcription_result.confidence_score,
                "model": transcription_result.model_used,
            },
        }

    def _correct_text(self, text: str) -> str:
        """
        Apply corrective pass using TranscriptCorrector
        (Simulated LLM + Terminology check).
        """
        return self.corrector.correct_transcript(text, context="therapy_session")

    def _check_therapeutic_alignment(self, text: str) -> Dict[str, Any]:
        """
        Evaluates therapeutic alignment (Pass 3).
        """
        # Use NeMo Evaluator for judged alignment
        # predictions are the transcript, references would be therapeutic gold standards
        # Here we use it in a 'zero-shot evaluation' mode if supported by the NIM
        try:
            # Dummy references for comparison if NIM requires pairing
            return self.evaluator.evaluate_therapeutic_alignment(
                predictions=[text],
                references=[
                    "Reflect empathy, use trauma-informed language, validate emotions."
                ],
            )
        except Exception as e:
            logger.error(f"Therapeutic alignment check failed: {e}")
            return {"score": 0.0, "status": "unknown"}


if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python transcript_quality_pipeline.py <audio_file>")
        sys.exit(1)

    audio_path = Path(sys.argv[1])
    pipeline = TranscriptQualityPipeline()
    result = pipeline.process_audio(audio_path)
    print(json.dumps(result, indent=2))
