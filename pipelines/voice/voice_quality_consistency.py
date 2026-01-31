import glob
import json
import logging
import os

import numpy as np

# Configuration
QUALITY_DIR = "data/voice_quality"
FEATURES_DIR = "data/voice"
CONSISTENCY_DIR = "data/voice_consistency"
LOG_FILE = "logs/voice_quality_consistency.log"
os.makedirs(CONSISTENCY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Scoring weights (tunable)
WEIGHTS = {
    "snr": 0.3,
    "loudness": 0.2,
    "silence": 0.1,
    "clipping": 0.1,
    "language": 0.1,
    "emotion_stability": 0.2,
}


def load_json_files(directory):
    files = glob.glob(os.path.join(directory, "*.json"))
    data = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            data[os.path.basename(f)] = json.load(fh)
    return data


def compute_emotion_stability(feature_segments):
    # Compute stddev of emotion scores across segments
    emotion_keys = [k for k in feature_segments[0] if k.startswith("emotion_")]
    stabilities = []
    for k in emotion_keys:
        vals = [seg.get(k, 0.0) for seg in feature_segments]
        stabilities.append(np.std(vals))
    # Lower stddev = more stable
    return 1.0 - np.mean(stabilities) if stabilities else 1.0


def score_file(quality_metrics, feature_segments):
    # Normalize and combine metrics
    snr = min(max(quality_metrics.get("snr", 0) / 30.0, 0), 1)
    loudness = min(max(quality_metrics.get("loudness", 0) / -16.0, 0), 1)
    silence = 1.0 - min(max(quality_metrics.get("silence_ratio", 0), 0), 1)
    clipping = 1.0 - min(max(quality_metrics.get("clipping_ratio", 0), 0), 1)
    language = 1.0 if quality_metrics.get("language", "en") == "en" else 0.0
    emotion_stability = compute_emotion_stability(feature_segments)
    score = (
        WEIGHTS["snr"] * snr
        + WEIGHTS["loudness"] * loudness
        + WEIGHTS["silence"] * silence
        + WEIGHTS["clipping"] * clipping
        + WEIGHTS["language"] * language
        + WEIGHTS["emotion_stability"] * emotion_stability
    )
    return {
        "snr": snr,
        "loudness": loudness,
        "silence": silence,
        "clipping": clipping,
        "language": language,
        "emotion_stability": emotion_stability,
        "composite_score": score,
    }


def main():
    quality_data = load_json_files(QUALITY_DIR)
    feature_data = load_json_files(FEATURES_DIR)
    results = []
    for fname, qmetrics in quality_data.items():
        base = os.path.splitext(fname)[0]
        feature_fname = f"{base}_features.json"
        feature_segments = feature_data.get(feature_fname, [])
        if not feature_segments:
            logging.warning(f"No feature data for {fname}")
            continue
        scores = score_file(qmetrics, feature_segments)
        result = {"file": fname, **scores}
        results.append(result)
    out_path = os.path.join(CONSISTENCY_DIR, "voice_quality_consistency.json")
    with open(out_path, encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(
        f"Voice quality and consistency assessment complete. Results saved to {out_path}"
    )


if __name__ == "__main__":
    main()
