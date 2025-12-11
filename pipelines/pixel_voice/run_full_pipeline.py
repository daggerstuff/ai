import logging
import os
import subprocess
import sys

LOG_FILE = "logs/run_full_pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

PIPELINE_STAGES = [
    ("Audio Quality Control", "pixel_voice/audio_quality_control.py"),
    ("Batch Transcription", "pixel_voice/batch_transcribe.py"),
    ("Transcription Quality Filtering", "pixel_voice/transcription_quality_filter.py"),
    ("Feature Extraction", "pixel_voice/feature_extraction.py"),
    ("Personality & Emotion Clustering", "pixel_voice/personality_emotion_clustering.py"),
    ("Dialogue Pair Construction", "pixel_voice/dialogue_pair_constructor.py"),
    ("Dialogue Pair Validation", "pixel_voice/dialogue_pair_validation.py"),
    ("Therapeutic Pair Generation", "pixel_voice/generate_therapeutic_pairs.py"),
    ("Voice Quality Consistency", "pixel_voice/voice_quality_consistency.py"),
    ("Voice Data Filtering/Optimization", "pixel_voice/voice_data_filtering.py"),
    ("Pipeline Reporting", "pixel_voice/pipeline_reporting.py"),
]


def run_stage(name, script):
    logging.info(f"Starting stage: {name} ({script})")
    try:
        result = subprocess.run(
            [sys.executable, script], check=True, capture_output=True, text=True
        )
        logging.info(f"Stage '{name}' completed successfully.")
        if result.stdout:
            logging.info(f"Output: {result.stdout}")
        if result.stderr:
            logging.warning(f"Stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Stage '{name}' failed: {e}\nStderr: {e.stderr}")
        sys.exit(1)


def main():
    for name, script in PIPELINE_STAGES:
        run_stage(name, script)
    logging.info("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
