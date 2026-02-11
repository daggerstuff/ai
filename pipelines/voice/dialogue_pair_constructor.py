import json
import logging
import os

# Configuration
CLUSTER_FILE = "data/voice_clusters/personality_emotion_clusters.json"
PAIR_DIR = "data/dialogue_pairs"
LOG_FILE = "logs/dialogue_pair_constructor.log"
os.makedirs(PAIR_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def load_segments(cluster_file):

    with open(cluster_file, encoding="utf-8") as f:
        segments = json.load(f)
    # Sort by file and start time for dialogue continuity
    segments.sort(key=lambda x: (x["file"], x["start"]))
    return segments


def construct_pairs(segments):
    pairs = []
    prev = None
    for seg in segments:
        if prev is not None and prev["file"] == seg["file"]:
            pair = {
                "file": seg["file"],
                "turn_1": {
                    "start": prev["start"],
                    "end": prev["end"],
                    "text": prev["text"],
                    "cluster": prev.get("cluster"),
                    "emotion": {k: v for k, v in prev.items() if k.startswith("emotion_")},
                    "sentiment_label": prev.get("sentiment_label"),
                    "sentiment_score": prev.get("sentiment_score"),
                    "outlier": prev.get("outlier"),
                },
                "turn_2": {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "cluster": seg.get("cluster"),
                    "emotion": {k: v for k, v in seg.items() if k.startswith("emotion_")},
                    "sentiment_label": seg.get("sentiment_label"),
                    "sentiment_score": seg.get("sentiment_score"),
                    "outlier": seg.get("outlier"),
                },
                "pair_metadata": {
                    "turn_1_cluster": prev.get("cluster"),
                    "turn_2_cluster": seg.get("cluster"),
                    "turn_1_outlier": prev.get("outlier"),
                    "turn_2_outlier": seg.get("outlier"),
                    "turn_1_sentiment": prev.get("sentiment_label"),
                    "turn_2_sentiment": seg.get("sentiment_label"),
                },
            }
            pairs.append(pair)
        prev = seg
    return pairs


def main():
    segments = load_segments(CLUSTER_FILE)
    pairs = construct_pairs(segments)
    out_path = os.path.join(PAIR_DIR, "dialogue_pairs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)
    logging.info(f"Constructed {len(pairs)} dialogue pairs. Saved to {out_path}")


if __name__ == "__main__":
    main()
