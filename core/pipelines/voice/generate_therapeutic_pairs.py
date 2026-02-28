import json
import logging
import os

# Configuration
VALIDATED_FILE = "data/dialogue_pairs/dialogue_pairs_validated.json"
THERAPEUTIC_DIR = "data/therapeutic_pairs"
LOG_FILE = "logs/generate_therapeutic_pairs.log"
os.makedirs(THERAPEUTIC_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

# Thresholds for therapeutic selection (can be tuned)
EMPATHY_THRESHOLD = 0.7
TOXICITY_THRESHOLD = 0.3


def is_therapeutic(pair):
    try:
        # Empathy: at least one turn with high empathy
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
        # Toxicity: both turns must be low toxicity
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
        # Appropriateness, naturalness, etc. can be added here
        return (emp1 >= EMPATHY_THRESHOLD or emp2 >= EMPATHY_THRESHOLD) and (
            tox1 <= TOXICITY_THRESHOLD and tox2 <= TOXICITY_THRESHOLD
        )
    except Exception as e:
        logging.error(f"Therapeutic filter error: {e}")
        return False


def main():
    with open(VALIDATED_FILE, encoding="utf-8") as f:
        pairs = json.load(f)
    therapeutic_pairs = [pair for pair in pairs if is_therapeutic(pair)]
    out_path = os.path.join(THERAPEUTIC_DIR, "therapeutic_pairs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(therapeutic_pairs, f, indent=2)
    logging.info(f"Selected {len(therapeutic_pairs)} therapeutic pairs. " f"Saved to {out_path}")


if __name__ == "__main__":
    main()
