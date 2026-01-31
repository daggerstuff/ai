import glob
import json
import logging
import os

MIN_TEXT_LENGTH = 10

# Configuration
VALIDATION_DIR = "data/dialogue_validation"
DIALOGUE_DIR = "data/dialogue_format"
OUTPUT_DIR = "data/therapeutic_pairs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def is_empathetic(turn):
    # Placeholder: Replace with advanced empathy detection
    text = turn.get("text", "").lower()
    return any(
        word in text for word in ["understand", "feel", "support", "empath", "care", "listen"]
    )


def is_appropriate(turn):
    # Placeholder: Replace with advanced appropriateness checks
    return len(turn.get("text", "")) > MIN_TEXT_LENGTH


def generate_pairs(dialogue):
    pairs = []
    for i in range(len(dialogue) - 1):
        prompt = dialogue[i]
        response = dialogue[i + 1]
        if is_empathetic(response) and is_appropriate(response):
            pairs.append(
                {
                    "prompt": prompt["text"],
                    "response": response["text"],
                    "prompt_meta": prompt.get("personality_metadata", {}),
                    "response_meta": response.get("personality_metadata", {}),
                }
            )
    return pairs


def main():
    dialogue_files = glob.glob(os.path.join(DIALOGUE_DIR, "*.jsonl"))
    for dfile in dialogue_files:
        base = os.path.splitext(os.path.basename(dfile))[0]
        with open(dfile, encoding="utf-8") as f:
            dialogue = [json.loads(line) for line in f]
        pairs = generate_pairs(dialogue)
        output_path = os.path.join(OUTPUT_DIR, f"{base}_pairs.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        logging.info(f"Therapeutic pairs written: {output_path} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
