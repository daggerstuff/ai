import os
import json
import logging
import glob
import numpy as np

# Configuration
CONSISTENCY_FILE = "data/voice_consistency/voice_quality_consistency.json"
THERAPEUTIC_FILE = "data/therapeutic_pairs/therapeutic_pairs.json"
OPTIMIZED_DIR = "data/voice_optimized"
LOG_FILE = "logs/voice_data_filtering.log"
os.makedirs(OPTIMIZED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Filtering thresholds and optimization parameters
MIN_COMPOSITE_SCORE = 0.7
MIN_EMPATHY_SCORE = 0.7
MAX_TOXICITY_SCORE = 0.3
DIVERSITY_CLUSTER_COUNT = 5  # Minimum number of clusters to cover


def load_json(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def select_diverse_pairs(pairs, cluster_key="turn_1_cluster"):
    # Select pairs to maximize cluster diversity
    selected = []
    seen_clusters = set()
    for pair in pairs:
        cluster = pair["pair_metadata"].get(cluster_key)
        if cluster not in seen_clusters:
            selected.append(pair)
            seen_clusters.add(cluster)
        if len(seen_clusters) >= DIVERSITY_CLUSTER_COUNT:
            break
    return selected


def optimize_dataset(consistency_data, therapeutic_pairs):
    # Filter by composite score
    high_quality_files = {
        d["file"] for d in consistency_data if d["composite_score"] >= MIN_COMPOSITE_SCORE
    }
    # Filter therapeutic pairs by empathy/toxicity
    filtered_pairs = []
    for pair in therapeutic_pairs:
        emp1 = (
            pair["validation"]["empathy_turn_1"][0]["score"]
            if pair["validation"].get("empathy_turn_1")
            else 0
        )
        emp2 = (
            pair["validation"]["empathy_turn_2"][0]["score"]
            if pair["validation"].get("empathy_turn_2")
            else 0
        )
        tox1 = (
            pair["validation"]["toxicity_turn_1"][0]["score"]
            if pair["validation"].get("toxicity_turn_1")
            else 1
        )
        tox2 = (
            pair["validation"]["toxicity_turn_2"][0]["score"]
            if pair["validation"].get("toxicity_turn_2")
            else 1
        )
        file = pair.get("file")
        if (
            (emp1 >= MIN_EMPATHY_SCORE or emp2 >= MIN_EMPATHY_SCORE)
            and (tox1 <= MAX_TOXICITY_SCORE and tox2 <= MAX_TOXICITY_SCORE)
            and (file in high_quality_files)
        ):
            filtered_pairs.append(pair)
    # Select for diversity
    diverse_pairs = select_diverse_pairs(filtered_pairs)
    return diverse_pairs


def main():
    consistency_data = load_json(CONSISTENCY_FILE)
    therapeutic_pairs = load_json(THERAPEUTIC_FILE)
    optimized = optimize_dataset(consistency_data, therapeutic_pairs)
    out_path = os.path.join(OPTIMIZED_DIR, "voice_optimized_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(optimized, f, indent=2)
    logging.info(f"Optimized dataset created with {len(optimized)} pairs. Saved to {out_path}")


if __name__ == "__main__":
    main()
