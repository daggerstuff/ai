import os
import json
import glob

# Configuration
INPUT_DIR = "data/dialogue_format"
OUTPUT_DIR = "data/dialogue_validation"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def validate_turn(turn):
    # Placeholder: Replace with advanced checks as needed
    text = turn.get("text", "")
    meta = turn.get("personality_metadata", {})
    # Simple heuristics
    is_authentic = meta.get("has_emotion_word", False) and meta.get("avg_word_length", 0) > 3.5
    is_natural = len(text.split()) > 3 and text[0].isupper() and text[-1] in ".!?"
    return {
        "start": turn.get("start"),
        "end": turn.get("end"),
        "is_authentic": is_authentic,
        "is_natural": is_natural,
        "text": text,
    }


def process_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f:
        dialogue = [json.loads(line) for line in f]
    results = []
    for turn in dialogue:
        result = validate_turn(turn)
        results.append(result)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Validation results written: {output_path}")


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*.jsonl"))
    if not files:
        print(f"No dialogue files found in {INPUT_DIR}")
        return
    for input_path in files:
        base = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{base}_validation.json")
        process_file(input_path, output_path)


if __name__ == "__main__":
    main()
