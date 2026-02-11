import glob
import logging
import os
import subprocess
import sys
from pathlib import Path

# Configuration
SEGMENT_DIR = "data/voice_segments"
TRANSCRIPT_DIR = "data/voice_transcripts"
LOG_FILE = "logs/batch_transcribe.log"
WHISPERX_MODEL = "large-v2"  # or "medium", "base", etc.
LANGUAGE = "en"  # or "auto"
BATCH_SIZE = 4  # Number of files to process in parallel

os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def transcribe_with_whisperx(
    audio_path,
    output_dir,
    model=WHISPERX_MODEL,
    language=LANGUAGE,
):
    try:
        out_path = os.path.join(output_dir, f"{Path(audio_path).stem}_whisperx.json")
        if os.path.exists(out_path):
            logging.info(f"Transcript already exists for {audio_path}, skipping.")
            return
        cmd = [
            "whisperx",
            audio_path,
            "--output_dir",
            output_dir,
            "--output_format",
            "json",
            "--model",
            model,
            "--language",
            language,
            "--diarize",
            "--compute_type",
            "float16",
        ]
        logging.info(f"Transcribing: {audio_path}")
        subprocess.run(cmd, check=True)
        logging.info(f"Completed: {audio_path}")
    except Exception as e:
        logging.error(f"Error transcribing {audio_path}: {e}")


def main():

    audio_files = glob.glob(os.path.join(SEGMENT_DIR, "*.wav"))
    if not audio_files:
        logging.error(f"No audio files found in {SEGMENT_DIR}")
        sys.exit(1)

    from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        for audio_path in audio_files:
            executor.submit(transcribe_with_whisperx, audio_path, TRANSCRIPT_DIR)
    logging.info("All batch transcriptions complete.")


if __name__ == "__main__":
    main()
