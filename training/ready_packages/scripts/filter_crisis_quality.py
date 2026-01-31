#!/usr/bin/env python3
"""
Filter Crisis Detection Dataset for Quality
Removes samples where the therapist's response is misaligned with the client's input.
"""

import json
from pathlib import Path

from tqdm import tqdm


def filter_crisis_dataset(input_file, output_file):
    input_path = Path(input_file)
    output_path = Path(output_file)

    kept_count = 0
    removed_count = 0

    # Common crisis keywords
    crisis_keywords = [
        "suicide",
        "dark place",
        "kill myself",
        "end it",
        "hurt myself",
        "don't want to live",
        "die",
        "death",
    ]

    with (
        open(input_path, "r", encoding="utf-8", errors="replace") as f,
        open(output_path, "w", encoding="utf-8") as out,
    ):
        for line in tqdm(f, desc="Filtering crisis dataset"):
            try:
                data = json.loads(line)
                messages = data.get("conversation", [])
                if not messages:
                    continue

                client_text = messages[0].get("content", "").lower()
                therapist_text = (
                    messages[1].get("content", "").lower() if len(messages) > 1 else ""
                )

                # Heuristic: If therapist mentions suicide but client doesn't mention any crisis keywords
                if (
                    "suicide" in therapist_text
                    or "help you through this moment" in therapist_text
                ):
                    if not any(kw in client_text for kw in crisis_keywords):
                        removed_count += 1
                        continue

                out.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept_count += 1
            except Exception:
                continue

    print("✅ Filtered crisis dataset:")
    print(f"   Kept: {kept_count}")
    print(f"   Removed: {removed_count}")


if __name__ == "__main__":
    # Example for the current partial file
    # Path depends on what sync_expanded_library.sh did
    input_f = "ai/training_ready/data/generated/edge_case_expanded/crisis_detection_conversations.jsonl.de83F1d6"
    output_f = "ai/training_ready/data/generated/edge_case_expanded/crisis_detection_cleaned.jsonl"
    if Path(input_f).exists():
        filter_crisis_dataset(input_f, output_f)
