#!/usr/bin/env python3
import json
import os
from pathlib import Path

import requests

# Set up to use the user's Ollama instance
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.pixelated.love")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "ornith:9b")

def evaluate_pair(instruction: str, output: str) -> int:
    system_prompt = (
        "You are an expert dataset curator for a therapeutic LLM. "
        "Review the following instruction (client statement) and output (therapist response). "
        "Does this make sense as a coherent, conversational therapeutic exchange for training a model, "
        "or is it disjointed/non-sensical because it was just extracted blindly from a book? "
        "Score the coherence and training viability strictly from 1 to 5. "
        "Output ONLY the integer score."
    )
    user_content = f"Instruction: {instruction}\nOutput: {output}"

    url = f"{OLLAMA_HOST.rstrip('/')}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"System: {system_prompt}\n\nUser: {user_content}",
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        if response.status_code == 200:
            score_text = response.json().get("response", "").strip()
            # Extract just the digit
            digits = [c for c in score_text if c.isdigit()]
            if digits:
                return int(digits[0])
    except Exception as e:
        print(f"Error querying LLM: {e}")
    return 0

def format_chatml(instruction: str, output: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output}
        ]
    }

def main():
    base_dir = Path("ai/training/output/books/clean")
    if not base_dir.exists():
        print("Clean directory not found.")
        return

    # Let's just process the first 10 pairs from one file as a test run
    test_file = next(base_dir.glob("*.jsonl"), None)
    if not test_file:
        print("No jsonl files found.")
        return

    print(f"Running LLM Evaluator on a sample of 10 pairs from: {test_file.name}")
    print("=" * 60)

    with open(test_file, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()][:10]

    good_pairs = []
    bad_pairs = []

    for i, line in enumerate(lines):
        try:
            pair = json.loads(line)
        except:
            continue

        inst = pair.get("instruction", "")
        out = pair.get("output", "")

        score = evaluate_pair(inst, out)
        print(f"Pair {i+1}: Score {score}/5")
        print(f"  User: {inst[:60]}...")
        print(f"  Asst: {out[:60]}...")

        if score >= 4:
            good_pairs.append(format_chatml(inst, out))
        else:
            bad_pairs.append(pair)

    print("=" * 60)
    print("Total evaluated: 10")
    print(f"Passed (Score 4+): {len(good_pairs)}")
    print(f"Failed (Score < 4): {len(bad_pairs)}")

    if bad_pairs:
        print("\nExample of a FAILED pair:")
        print(f"Instruction: {bad_pairs[0]['instruction']}")
        print(f"Output: {bad_pairs[0]['output']}")

if __name__ == "__main__":
    main()
