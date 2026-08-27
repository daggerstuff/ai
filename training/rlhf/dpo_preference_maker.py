"""
Pixelated Empathy: DPO Preference Maker (Anti-Echo / Human-Pivot)

This script generates preference pairs for RLHF/DPO.
Chosen: Human-like, direct, parallel reflection, challenging.
Rejected: Validating echo, 'I hear that...', AI-style mirroring.
"""

import json
from pathlib import Path


def create_dpo_pair(
    patient_query: str, ai_style_response: str, human_pivot_response: str
):
    """
    Creates a DPO data point.
    """
    return {
        "prompt": patient_query,
        "chosen": human_pivot_response,
        "rejected": ai_style_response,
        "metadata": {"type": "anti_echo_shift", "principle": "parallel_reflection"},
    }


# Dataset Generator Example
def generate_human_pivot_dataset():
    data = []

    append_dpo_pair_to_list(
        "I finally left him. I should be happy, I've wanted this for years. "
        "But I just feel... hollow.",
        "I hear that you're feeling hollow after leaving him. "
        "It's completely normal to feel a sense of loss after a big change.",
        "I was thinking about the way shadows stretch in the winter—they’re long "
        "and thin, and they don't provide any warmth, but when they’re gone, "
        "the ground just looks featureless. Is it the person you miss, "
        "or just the shape of the life you had?",
        data,
    )
    append_dpo_pair_to_list(
        "I'm just a failure at everything. Every job, every relationship.",
        "It sounds like you're feeling really down on yourself right now. "
        "I can see you've had a lot of difficult experiences.",
        "Why are you trying so hard to convince me that you're a failure? "
        "It's like you're building a cage and asking me lock the door. "
        "Tell me about the first time you decided this was your 'story'.",
        data,
    )
    return data


def append_dpo_pair_to_list(query, rejected_response, chosen_response, dataset_list):
    dataset_list.append(create_dpo_pair(query, rejected_response, chosen_response))


def merge_preference_files(
    input_paths: list[Path], output_path: Path
) -> int:
    """Merge multiple JSONL preference files into one.

    Each input file must contain lines with 'prompt', 'chosen', 'rejected' fields.
    The merged file preserves all records and adds a 'source_file' metadata field.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(output_path, "w") as out_f:
        for in_path in input_paths:
            if not in_path.exists():
                print(f"WARNING: {in_path} not found, skipping")
                continue
            count = 0
            with open(in_path) as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    record.setdefault("metadata", {})["source_file"] = str(in_path)
                    out_f.write(json.dumps(record) + "\n")
                    count += 1
            total += count
            print(f"  Merged {count} pairs from {in_path}")
    print(f"Total merged: {total} pairs → {output_path}")
    return total


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Generate and merge DPO preference pairs for therapeutic AI alignment.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing JSONL preference files (skip generation).",
    )
    parser.add_argument(
        "--extra-prefs",
        type=str,
        nargs="+",
        default=None,
        help="Additional JSONL preference files to merge with anti-echo prefs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai/training/rlhf/combined_prefs.jsonl",
        help="Output path for merged preference file.",
    )
    args = parser.parse_args()

    if not args.merge_only:
        # Generate anti-echo preferences
        dataset = generate_human_pivot_dataset()
        anti_echo_path = Path("ai/training/rlhf/anti_echo_prefs.jsonl")
        anti_echo_path.parent.mkdir(parents=True, exist_ok=True)
        with open(anti_echo_path, "w") as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
        print(f"Generated {len(dataset)} anti-echo DPO pairs → {anti_echo_path}")

    if args.extra_prefs:
        extra_paths = [Path(p) for p in args.extra_prefs]
        if not args.merge_only:
            merge_preference_files(
                [Path("ai/training/rlhf/anti_echo_prefs.jsonl"), *extra_paths],
                Path(args.output),
            )
        else:
            merge_preference_files(extra_paths, Path(args.output))
    elif not args.merge_only:
        # Default: just write anti-echo (backward compatible)
        output_path = Path("ai/training/rlhf/anti_echo_prefs.jsonl")
        print(f"Generated {len(dataset)} anti-echo DPO pairs.")
