import os
import json
import glob

# Configuration
INPUT_DIR = "data/voice_transcripts"
OUTPUT_DIR = "data/voice_transcripts_filtered"
CONFIDENCE_THRESHOLD = 0.85

os.makedirs(OUTPUT_DIR, exist_ok=True)


def filter_segments(transcript, threshold):
    # Assumes transcript is a dict with a "segments" key (Faster Whisper/WhisperX format)
    filtered = []
    for seg in transcript.get("segments", []):
        if seg.get("avg_logprob", 0) >= threshold or seg.get("confidence", 1) >= threshold:
            filtered.append(seg)
    return filtered


def process_file(input_path, output_path, threshold):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    filtered_segments = filter_segments(data, threshold)
    filtered_data = dict(data)
    filtered_data["segments"] = filtered_segments
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filtered_data, f, indent=2)
    print(f"Filtered {input_path} -> {output_path} ({len(filtered_segments)} segments kept)")


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not files:
        print(f"No transcript files found in {INPUT_DIR}")
        return
    for input_path in files:
        base = os.path.basename(input_path)
        output_path = os.path.join(OUTPUT_DIR, base)
        process_file(input_path, output_path, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    main()
