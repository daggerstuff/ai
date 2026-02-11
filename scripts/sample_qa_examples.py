#!/usr/bin/env python3
"""
Sample random Q/A examples from the cleaned dataset
"""

import json
import random
from pathlib import Path

def sample_random_examples(dataset_file: Path, num_samples: int = 5):
    """Sample random Q/A examples from dataset"""
    with open(dataset_file, 'r', encoding='utf-8') as f:
        conversations = json.load(f)
    
    # Get random samples from different parts of dataset
    total = len(conversations)
    indices = [
        random.randint(0, total//5),           # Early part
        random.randint(total//5, 2*total//5),  # Early-mid
        random.randint(2*total//5, 3*total//5), # Middle
        random.randint(3*total//5, 4*total//5), # Mid-late
        random.randint(4*total//5, total-1)     # Late part
    ]
    
    print("=== 5 RANDOM Q/A EXAMPLES FROM DATASET ===\n")
    
    for i, idx in enumerate(indices, 1):
        conv = conversations[idx]
        
        print(f"**Example {i}** (Index: {idx:,}/{total:,})")
        print(f"**Style**: {conv['style']} (Expert {conv['expert_id']})")
        print(f"**Source**: {conv.get('source', 'unknown')}")
        print(f"**Quality**: {conv.get('quality', 'N/A')}")
        print()
        print(f"**Q**: {conv['conversations'][0]['value']}")
        print()
        print(f"**A**: {conv['conversations'][1]['value'][:500]}{'...' if len(conv['conversations'][1]['value']) > 500 else ''}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    dataset_file = Path("/root/pixelated/ai/data/lightning_h100_complete/train.json")
    sample_random_examples(dataset_file)
