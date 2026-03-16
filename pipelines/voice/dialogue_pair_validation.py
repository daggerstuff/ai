import json
import logging
import os

from transformers.pipelines import pipeline

# Configuration
PAIR_FILE = "data/dialogue_pairs/dialogue_pairs.json"
VALIDATED_DIR = "data/dialogue_pairs"
LOG_FILE = "logs/dialogue_pair_validation.log"
os.makedirs(VALIDATED_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)


logging.basicConfig(
    filename=LOG_FILE, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# Placeholder: Replace with production-grade/fine-tuned models as available
empathy_classifier = pipeline(
    "text-classification",
    model="mrm8488/t5-base-finetuned-empathy",
    top_k=None,
)
toxicity_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None,
)
# TODO: Add naturalness, authenticity, appropriateness models


def validate_pair(pair):
    turn_1 = pair["turn_1"]["text"]
    turn_2 = pair["turn_2"]["text"]
    validation = {}
    try:
        validation["empathy_turn_1"] = empathy_classifier(turn_1)
        validation["empathy_turn_2"] = empathy_classifier(turn_2)
    except Exception as e:
        logging.error(f"Empathy validation failed: {e}")
    try:
        validation["toxicity_turn_1"] = toxicity_classifier(turn_1)
        validation["toxicity_turn_2"] = toxicity_classifier(turn_2)
    except Exception as e:
        logging.error(f"Toxicity validation failed: {e}")
    # TODO: Add naturalness, appropriateness, authenticity scoring
    return validation


def main():
    with open(PAIR_FILE, encoding="utf-8") as f:
        pairs = json.load(f)
    validated_pairs = []
    for pair in pairs:
        validation = validate_pair(pair)
        pair["validation"] = validation
        validated_pairs.append(pair)
    out_path = os.path.join(VALIDATED_DIR, "dialogue_pairs_validated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(validated_pairs, f, indent=2)
    logging.info(f"Validated {len(validated_pairs)} dialogue pairs. Saved to {out_path}")


if __name__ == "__main__":
    main()
