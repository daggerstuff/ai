"""
Pixelated Empathy: DPO Preference Maker (Anti-Echo / Human-Pivot)

This script generates preference pairs for RLHF/DPO.
Chosen: Human-like, direct, parallel reflection, challenging.
Rejected: Validating echo, 'I hear that...', AI-style mirroring.
"""

import json
from pathlib import Path


def create_dpo_pair(patient_query: str, ai_style_response: str, human_pivot_response: str):
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

    _extracted_from_generate_human_pivot_dataset_5(
        "I finally left him. I should be happy, I've wanted this for years. But I just feel... hollow.",
        "I hear that you're feeling hollow after leaving him. It's completely normal to feel a sense of loss after a big change.",
        "I was thinking about the way shadows stretch in the winter—they’re long and thin, and they don't provide any warmth, but when they’re gone, the ground just looks featureless. Is it the person you miss, or just the shape of the life you had?",
        data,
    )
    _extracted_from_generate_human_pivot_dataset_5(
        "I'm just a failure at everything. Every job, every relationship.",
        "It sounds like you're feeling really down on yourself right now. I can see you've had a lot of difficult experiences.",
        "Why are you trying so hard to convince me that you're a failure? It's like you're building a cage and asking me to lock the door. Tell me about the first time you decided this was your 'story'.",
        data,
    )
    return data


# TODO Rename this here and in `generate_human_pivot_dataset`
def _extracted_from_generate_human_pivot_dataset_5(arg0, arg1, arg2, data):
    # CASE 1: The Hollow Test
    query_1 = arg0
    rejected_1 = arg1
    chosen_1 = arg2
    data.append(create_dpo_pair(query_1, rejected_1, chosen_1))


if __name__ == "__main__":
    dataset = generate_human_pivot_dataset()
    output_path = Path("ai/training/rlhf/anti_echo_prefs.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")
    print(f"Generated {len(dataset)} anti-echo DPO pairs.")
