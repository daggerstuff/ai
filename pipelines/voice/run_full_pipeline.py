import logging
import os
import subprocess
import sys

LOG_FILE = "logs/run_full_pipeline.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PIPELINE_STAGES = [
    ("Audio Quality Control", os.path.join(SCRIPT_DIR, "audio_quality_control.py")),
    ("Batch Transcription", os.path.join(SCRIPT_DIR, "batch_transcribe.py")),
    ("Transcription Quality Filtering", os.path.join(SCRIPT_DIR, "transcription_quality_filter.py")),
    ("Feature Extraction", os.path.join(SCRIPT_DIR, "feature_extraction.py")),
    ("Personality & Emotion Clustering", os.path.join(SCRIPT_DIR, "personality_emotion_clustering.py")),
    ("Dialogue Pair Construction", os.path.join(SCRIPT_DIR, "dialogue_pair_constructor.py")),
    ("Dialogue Pair Validation", os.path.join(SCRIPT_DIR, "dialogue_pair_validation.py")),
    ("Therapeutic Pair Generation", os.path.join(SCRIPT_DIR, "generate_therapeutic_pairs.py")),
    ("Voice Quality Consistency", os.path.join(SCRIPT_DIR, "voice_quality_consistency.py")),
    ("Voice Data Filtering/Optimization", os.path.join(SCRIPT_DIR, "voice_data_filtering.py")),
    ("Pipeline Reporting", os.path.join(SCRIPT_DIR, "pipeline_reporting.py")),
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
        print(f"Error in stage '{name}'. Check logs for details.")
        sys.exit(1)


def main():
    for name, script in PIPELINE_STAGES:
        run_stage(name, script)
    logging.info("Full pipeline completed successfully.")
    print("Full pipeline completed successfully.")


if __name__ == "__main__":
    main()
