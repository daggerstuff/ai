import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score

# Configuration
FEATURES_DIR = "data/voice"
CLUSTER_DIR = "data/voice_clusters"
LOG_FILE = "logs/personality_emotion_clustering.log"
os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def load_features(features_dir):
    feature_files = glob.glob(os.path.join(features_dir, "*_features.json"))
    all_features = []
    meta = []
    for fpath in feature_files:
        with open(fpath, encoding="utf-8") as f:
            feats = json.load(f)
            for seg in feats:
                # Flatten and collect relevant features
                entry = {
                    "file": Path(fpath).stem,
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "text": seg.get("text"),
                    "length": seg.get("length"),
                    "num_words": seg.get("num_words"),
                    "avg_word_length": seg.get("avg_word_length"),
                }
                # Extract emotion and sentiment scores
                emotion = seg.get("emotion")
                if isinstance(emotion, list) and len(emotion) > 0:
                    for emo in emotion:
                        entry[f"emotion_{emo['label']}"] = emo["score"]
                sentiment = seg.get("sentiment")
                if isinstance(sentiment, list) and len(sentiment) > 0:
                    entry["sentiment_label"] = sentiment[0]["label"]
                    entry["sentiment_score"] = sentiment[0]["score"]
                all_features.append(entry)
                meta.append(
                    {
                        "file": Path(fpath).stem,
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text"),
                    }
                )
    return all_features, meta


def feature_matrix(features, keys):
    x_data = []
    for f in features:
        row = []
        for k in keys:
            row.append(f.get(k, 0.0))
        x_data.append(row)
    return np.array(x_data)


def cluster_features(x_data, method="kmeans", n_clusters=5):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        labels = model.fit_predict(x_data)
        score = silhouette_score(x_data, labels) if len(set(labels)) > 1 else None
        return labels, model, score
    if method == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(x_data)
        score = silhouette_score(x_data, labels) if len(set(labels)) > 1 else None
        return labels, model, score
    raise ValueError("Unknown clustering method")


def detect_outliers(x_data):
    iso = IsolationForest(contamination=0.05, random_state=42)
    return iso.fit_predict(x_data)


def drift_detection(x_data, bins=20):
    # Simple drift: compare histograms of first and second halves
    mid = len(x_data) // 2
    if mid == 0:
        return None
    hist1, _ = np.histogram(
        x_data[:mid], bins=bins, range=(np.min(x_data), np.max(x_data)), density=True
    )
    hist2, _ = np.histogram(
        x_data[mid:], bins=bins, range=(np.min(x_data), np.max(x_data)), density=True
    )
    return entropy(hist1 + 1e-8, hist2 + 1e-8)


def main():
    features, _ = load_features(FEATURES_DIR)
    # Select numeric features for clustering
    emotion_keys = [k for k in features[0] if k.startswith("emotion_")]
    cluster_keys = ["length", "num_words", "avg_word_length", *emotion_keys]
    x_data = feature_matrix(features, cluster_keys)
    # Clustering
    labels, _, sil_score = cluster_features(x_data, method="kmeans", n_clusters=5)
    # Outlier detection
    outlier_labels = detect_outliers(x_data)
    # Drift detection (on emotion features)
    drift = drift_detection(x_data[:, -len(emotion_keys) :]) if emotion_keys else None
    # Save results
    results = []
    for i, f in enumerate(features):
        res = dict(f)
        res["cluster"] = int(labels[i])
        res["outlier"] = int(outlier_labels[i] == -1)
        results.append(res)
    out_path = os.path.join(CLUSTER_DIR, "personality_emotion_clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logging.info(
        f"Clustering complete. Silhouette score: {sil_score}. Drift: {drift}. Results saved to {out_path}"
    )


if __name__ == "__main__":
    main()
