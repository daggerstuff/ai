import os
import json
import glob
import logging
from pathlib import Path

# Configuration
TRANSCRIPT_DIR = "data/voice_transcripts"
FILTERED_DIR = "data/voice_transcripts_filtered"
REPORTS_DIR = "data/voice_transcript_reports"
LOG_FILE = "logs/transcription_quality_filter.log"
CONFIDENCE_THRESHOLD = 0.85

os.makedirs(FILTERED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def filter_segments(segments, threshold):
    filtered = [
        seg
        for seg in segments
        if seg.get("avg_logprob", 0) >= threshold or seg.get("confidence", 1) >= threshold
    ]
    return filtered


def process_file(input_path, filtered_dir, report_dir, threshold):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    filtered_segments = filter_segments(segments, threshold)
    filtered_data = dict(data)
    filtered_data["segments"] = filtered_segments

    base = Path(input_path).stem
    filtered_path = os.path.join(filtered_dir, f"{base}_filtered.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)

    report = {
        "file": input_path,
        "total_segments": len(segments),
        "filtered_segments": len(filtered_segments),
        "confidence_threshold": threshold,
        "filtered_ratio": len(filtered_segments) / max(1, len(segments)),
    }
    report_path = os.path.join(report_dir, f"{base}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logging.info(f"Filtered {input_path}: {len(filtered_segments)}/{len(segments)} segments kept")


def main():
    transcript_files = glob.glob(os.path.join(TRANSCRIPT_DIR, "*.json"))
    for tfile in transcript_files:
        try:
            process_file(tfile, FILTERED_DIR, REPORTS_DIR, CONFIDENCE_THRESHOLD)
        except Exception as e:
            logging.error(f"Error processing {tfile}: {e}")


if __name__ == "__main__":
    main()
