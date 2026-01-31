import os
import json
import glob

# Configuration
INPUT_DIR = "data/voice_transcripts_filtered"
OUTPUT_DIR = "data/personality_markers"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_markers_from_segment(segment):
    # Placeholder: Replace with real NLP/ML feature extraction
    text = segment.get("text", "")
    features = {
        "length": len(text),
        "num_words": len(text.split()),
        "has_emotion_word": any(
            word in text.lower() for word in ["happy", "sad", "angry", "excited", "calm"]
        ),
        "avg_word_length": sum(len(w) for w in text.split()) / max(1, len(text.split())),
        # Add more sophisticated features as needed
    }
    return features


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    markers = []
    for seg in data.get("segments", []):
        features = extract_markers_from_segment(seg)
        features["start"] = seg.get("start")
        features["end"] = seg.get("end")
        features["text"] = seg.get("text")
        markers.append(features)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(markers, f, indent=2)
    print(f"Extracted markers: {input_path} -> {output_path}")


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.json"))
    if not files:
        print(f"No filtered transcript files found in {INPUT_DIR}")
        return
    for input_path in files:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}_markers.json")
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()
