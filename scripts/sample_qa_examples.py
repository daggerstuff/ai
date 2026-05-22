#!/usr/bin/env python3
"""
Sample random Q/A examples from the cleaned dataset
"""

import json
import random
from pathlib import Path


def sample_random_examples(dataset_file: Path, _num_samples: int = 5):
    """Sample random Q/A examples from dataset"""
    with open(dataset_file, encoding="utf-8") as f:
        conversations = json.load(f)

    # Get random samples from different parts of dataset
    total = len(conversations)
    indices = [
        random.randint(0, total // 5),  # Early part
        random.randint(total // 5, 2 * total // 5),  # Early-mid
        random.randint(2 * total // 5, 3 * total // 5),  # Middle
        random.randint(3 * total // 5, 4 * total // 5),  # Mid-late
        random.randint(4 * total // 5, total - 1),  # Late part
    ]

    for _i, idx in enumerate(indices, 1):
        conversations[idx]


if __name__ == "__main__":
    dataset_file = Path("/root/pixelated/ai/data/lightning_h100_complete/train.json")
    sample_random_examples(dataset_file)
