import os
import json
import glob
import numpy as np

# Configuration
MARKERS_DIR = "data/personality_markers"
OUTPUT_DIR = "data/personality_consistency"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def aggregate_feature(markers, feature):
    values = [m.get(feature, 0) for m in markers if feature in m]
    return np.mean(values), np.std(values)


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        markers = json.load(f)
    features = ["length", "num_words", "avg_word_length"]
    consistency = {}
    for feat in features:
        mean, std = aggregate_feature(markers, feat)
        consistency[feat] = {"mean": mean, "std": std}
    # Emotional word presence consistency
    emotion_presence = [m.get("has_emotion_word", False) for m in markers]
    consistency["has_emotion_word"] = {
        "fraction": sum(emotion_presence) / len(emotion_presence) if emotion_presence else 0
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(consistency, f, indent=2)
    print(f"Personality consistency written: {output_path}")


def main():
    marker_files = glob.glob(os.path.join(MARKERS_DIR, "*_markers.json"))
    for mfile in marker_files:
        base = os.path.splitext(os.path.basename(mfile))[0].replace("_markers", "")
        output_path = os.path.join(OUTPUT_DIR, f"{base}_consistency.json")
        process_file(mfile, output_path)


if __name__ == "__main__":
    main()
