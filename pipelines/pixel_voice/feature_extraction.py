import glob
import json
import logging
import os
from pathlib import Path

from transformers.pipelines import pipeline  # type: ignore[reportPrivateImportUsage]

# Configuration
FILTERED_DIR = "data/voice_transcripts_filtered"
FEATURES_DIR = "data/voice_features"
LOG_FILE = "logs/feature_extraction.log"

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Load ML/NLP pipelines (can be swapped for custom/fine-tuned models)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)  # type: ignore
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # type: ignore
# Placeholder for empathy detection, personality, etc.

# Debug log added to validate input type and value


def extract_features(segment):
    import logging

    logging.debug(f"extract_features called with segment: {segment} (type: {type(segment)})")
    text = segment.get("text", "")
    features = {
        "length": len(text),
        "num_words": len(text.split()),
        "avg_word_length": sum(len(w) for w in text.split()) / max(1, len(text.split())),
        "emotion": None,
        "sentiment": None,
        # Add more features as needed
    }
    try:
        logging.info(f"Running emotion classification for text: {text[:50]}")
        features["emotion"] = emotion_classifier(text)
    except Exception as e:
        logging.error(f"Emotion classification failed: {e}")
    try:
        logging.info(f"Running sentiment classification for text: {text[:50]}")
        features["sentiment"] = sentiment_pipeline(text)
    except Exception as e:
        logging.error(f"Sentiment classification failed: {e}")
    # TODO: Add empathy, personality, rhythm, etc.
    return features


def process_file(input_path, features_dir):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)
    segments = data.get("segments", [])
    features_list = []
    for seg in segments:
        feats = extract_features(seg)
        feats["start"] = seg.get("start")
        feats["end"] = seg.get("end")
        feats["text"] = seg.get("text")
        features_list.append(feats)
    base = Path(input_path).stem
    features_path = os.path.join(features_dir, f"{base}_features.json")
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(features_list, f, indent=2)
    logging.info(f"Extracted features: {input_path} -> {features_path}")


def main():
    transcript_files = glob.glob(os.path.join(FILTERED_DIR, "*.json"))
    for tfile in transcript_files:
        try:
            process_file(tfile, FEATURES_DIR)
        except Exception as e:
            logging.error(f"Error processing {tfile}: {e}")


if __name__ == "__main__":
    main()
