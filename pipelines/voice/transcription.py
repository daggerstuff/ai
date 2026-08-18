import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

try:
    import whisperx
    from whisperx.diarize import DiarizationPipeline
except ImportError:
    raise ImportError(
        "whisperx and faster-whisper are not included in the main project dependencies. "
        "To use the voice pipeline, install them in a separate virtual environment:\n"
        "  uv venv .venv-voice\n"
        "  uv pip install -r requirements-voice.txt"
    ) from None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("TranscriptionPipeline")


def _assign_roles_by_duration(result: dict[str, Any]) -> dict[str, str]:
    """Rank speakers by speaking time.

    The most talkative speaker is assigned the Therapist role, the second
    most talkative the Client role. Speakers that cannot be ranked fall back
    to their raw label so downstream logic never silently drops a turn.
    """
    durations: dict[str, float] = {}
    for segment in result.get("segments", []):
        speaker = segment.get("speaker")
        if not speaker:
            continue
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        durations[speaker] = durations.get(speaker, 0.0) + max(0.0, end - start)

    ranked = sorted(durations.items(), key=lambda kv: kv[1], reverse=True)
    role_map: dict[str, str] = {}
    if len(ranked) >= 1:
        role_map[ranked[0][0]] = "Therapist"
    if len(ranked) >= 2:
        role_map[ranked[1][0]] = "Client"
    return role_map


class DiarizedTranscriptionPipeline:
    def __init__(self, device=None, compute_type="float16", hf_token=None, batch_size=16):
        self.device = (
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.batch_size = batch_size

        logger.info(f"Loading WhisperX large-v2 model on {self.device}...")
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

        logger.info("Loading alignment model...")
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code="en", device=self.device
        )

        logger.info("Loading diarization model...")
        self.diarize_model = DiarizationPipeline(token=self.hf_token, device=self.device)

    def process_audio(self, audio_file: str, min_confidence: float = 0.80) -> list[dict[str, Any]]:
        logger.info(f"Transcribing {audio_file}...")
        audio = whisperx.load_audio(audio_file)

        # 1. Transcribe
        result = self.model.transcribe(audio, batch_size=self.batch_size)

        # 2. Align whisper output
        result = whisperx.align(
            result["segments"],
            self.model_a,
            self.metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # 3. Assign speaker labels
        logger.info(f"Diarizing {audio_file}...")
        diarize_segments = self.diarize_model(audio, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

        role_map = _assign_roles_by_duration(result)

        # 4. Filter and format output
        final_segments = []
        for segment in result["segments"]:
            # WhisperX can return words with individual confidences. We take the mean.
            words = segment.get("words", [])
            valid_words = [w for w in words if "score" in w]

            if not valid_words:
                continue

            avg_confidence = sum(w["score"] for w in valid_words) / len(valid_words)

            if avg_confidence < min_confidence:
                logger.debug(f"Discarding segment due to low confidence ({avg_confidence:.2f})")
                continue

            speaker = segment.get("speaker", "UNKNOWN")
            role = role_map.get(speaker, "UNKNOWN")

            final_segments.append(
                {
                    "start": segment["start"],
                    "end": segment["end"],
                    "role": role,
                    "text": segment["text"].strip(),
                    "confidence": avg_confidence,
                }
            )

        return final_segments


class TranscriptionOrchestrator:
    def __init__(self, input_dir="ai/data/segmented_audio", output_dir="ai/data/transcripts"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None

    def process_all(self):
        if self.pipeline is None:
            self.pipeline = DiarizedTranscriptionPipeline()

        audio_files = list(self.input_dir.glob("*.wav"))
        logger.info(f"Found {len(audio_files)} audio chunks to process.")

        for audio_file in audio_files:
            out_file = self.output_dir / f"{audio_file.stem}.json"
            if out_file.exists():
                logger.info(f"Skipping {audio_file}, already transcribed.")
                continue

            try:
                segments = self.pipeline.process_audio(str(audio_file))
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(segments, f, indent=2)
                logger.info(f"Saved {len(segments)} valid segments to {out_file}")
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")


if __name__ == "__main__":
    orchestrator = TranscriptionOrchestrator()
    orchestrator.process_all()
