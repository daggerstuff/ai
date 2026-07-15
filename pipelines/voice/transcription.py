import json
import logging
import os
from pathlib import Path
from typing import Any

import torch
import whisperx

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TranscriptionPipeline")

AI_ROOT = Path(__file__).resolve().parents[2]

class DiarizedTranscriptionPipeline:
    def __init__(self, device=None, compute_type="float16", hf_token=None, batch_size=16):
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = compute_type if self.device == "cuda" else "int8"
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.batch_size = batch_size

        logger.info(f"Loading WhisperX large-v2 model on {self.device}...")
        self.model = whisperx.load_model("large-v2", self.device, compute_type=self.compute_type)

        logger.info("Loading alignment model...")
        self.model_a, self.metadata = whisperx.load_align_model(language_code="en", device=self.device)

        logger.info("Loading diarization model...")
        self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)  # type: ignore

    def process_audio(self, audio_file: str, min_confidence: float = 0.80) -> list[dict[str, Any]]:
        logger.info(f"Transcribing {audio_file}...")
        audio = whisperx.load_audio(audio_file)  # type: ignore

        # 1. Transcribe
        result = self.model.transcribe(audio, batch_size=self.batch_size)

        # 2. Align whisper output
        result = whisperx.align(  # type: ignore
            result["segments"], self.model_a, self.metadata, audio, self.device, return_char_alignments=False
        )

        # 3. Assign speaker labels
        logger.info(f"Diarizing {audio_file}...")
        diarize_segments = self.diarize_model(audio, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)  # type: ignore

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
                logger.debug(f"Discarding segment due to low confidence ({avg_confidence:.2f}): {segment['text']}")
                continue

            speaker = segment.get("speaker", "UNKNOWN")

            # Map SPEAKER_00 to Therapist, SPEAKER_01 to Client
            # (Heuristic based on most talking time, often therapist leads)
            # In production, we can run a classifier to distinguish role. Here we just maintain separate tags.
            role = "Therapist" if speaker == "SPEAKER_00" else "Client" if speaker == "SPEAKER_01" else speaker

            final_segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "role": role,
                "text": segment["text"].strip(),
                "confidence": avg_confidence
            })

        return final_segments

class TranscriptionOrchestrator:
    def __init__(
        self, input_dir: str | Path | None = None, output_dir: str | Path | None = None
    ):
        self.input_dir = Path(input_dir) if input_dir else AI_ROOT / "data/segmented_audio"
        self.output_dir = Path(output_dir) if output_dir else AI_ROOT / "data/transcripts"
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
