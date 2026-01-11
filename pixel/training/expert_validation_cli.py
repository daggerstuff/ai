"""
Expert Validation Dataset CLI

Scales curation to 500–1000 examples, enforces diversity balancing, exports JSONL +
manifest, and registers a training manifest entry. No model required.

Usage:
  uv run python ai/pixel/training/expert_validation_cli.py \
    --input conversations.jsonl \
    --output data/expert_validation/expert_dataset.jsonl \
    --target-count 800 \
    --min-crisis 0.08 \
    --max-crisis 0.20

If --input is omitted, a synthetic sample is generated using build_sample_conversations().
"""
from __future__ import annotations

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List

from ai.dataset_pipeline.schemas.conversation_schema import Conversation
from ai.dataset_pipeline.training_manifest import (
    create_safety_aware_manifest,
)
from ai.pixel.training.expert_validation_dataset import (
    ExpertValidationDataset,
    ExpertValidationExample,
    ScenarioType,
    build_sample_conversations,
)


def read_conversations_jsonl(path: Path) -> List[Conversation]:
    conversations: List[Conversation] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            conversations.append(Conversation.from_dict(data))
    return conversations


def balance_examples(examples: List[ExpertValidationExample], target: int, min_crisis_ratio: float, max_crisis_ratio: float) -> List[ExpertValidationExample]:
    """Simple diversity balancing across scenarios with crisis ratio bounds."""
    if not examples:
        return []

    # Group by scenario
    groups: Dict[str, List[ExpertValidationExample]] = {}
    for ex in examples:
        groups.setdefault(ex.scenario.value, []).append(ex)

    # Start with round-robin selection across scenarios
    ordered_scenarios = sorted(groups.keys())
    selected: List[ExpertValidationExample] = []
    offsets = {k: 0 for k in ordered_scenarios}

    while len(selected) < target and any(offsets[k] < len(groups[k]) for k in ordered_scenarios):
        for k in ordered_scenarios:
            idx = offsets[k]
            if idx < len(groups[k]):
                selected.append(groups[k][idx])
                offsets[k] += 1
                if len(selected) >= target:
                    break

    # Adjust crisis ratio within bounds by sampling if needed
    def crisis_ratio(items: List[ExpertValidationExample]) -> float:
        if not items:
            return 0.0
        return sum(1 for ex in items if ex.is_crisis) / len(items)

    # If too few crisis examples, top up from crisis pool
    crisis_pool = groups.get(ScenarioType.CRISIS.value, [])
    if crisis_ratio(selected) < min_crisis_ratio and crisis_pool:
        needed = int(min_crisis_ratio * target) - int(crisis_ratio(selected) * len(selected))
        needed = max(0, min(needed, len(crisis_pool)))
        # Append non-duplicates
        existing_ids = {ex.example_id for ex in selected}
        for ex in crisis_pool:
            if needed <= 0:
                break
            if ex.example_id in existing_ids:
                continue
            selected.append(ex)
            existing_ids.add(ex.example_id)
            needed -= 1

    # If too many crisis examples, trim randomly down to max ratio, then refill with non-crisis
    if crisis_ratio(selected) > max_crisis_ratio:
        cap = int(max_crisis_ratio * target)
        non_crisis_pool = [ex for ex in examples if not ex.is_crisis]
        keep_non_crisis = [ex for ex in selected if not ex.is_crisis]
        keep_crisis = [ex for ex in selected if ex.is_crisis]
        random.shuffle(keep_crisis)
        keep_crisis = keep_crisis[: max(0, cap)]
        selected = keep_non_crisis + keep_crisis
        # Refill up to target from non-crisis pool avoiding duplicates
        existing_ids = {ex.example_id for ex in selected}
        random.shuffle(non_crisis_pool)
        for ex in non_crisis_pool:
            if len(selected) >= target:
                break
            if ex.example_id in existing_ids:
                continue
            selected.append(ex)
            existing_ids.add(ex.example_id)

    # Truncate to target
    return selected[:target]


def main() -> None:
    parser = argparse.ArgumentParser(description="Scale expert validation dataset curation")
    parser.add_argument("--input", type=str, default="", help="Path to input conversations JSONL (optional)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path for expert dataset")
    parser.add_argument("--target-count", type=int, default=800, help="Target number of curated examples (500–1000)")
    parser.add_argument("--min-crisis", type=float, default=0.08, help="Minimum crisis ratio (0–1)")
    parser.add_argument("--max-crisis", type=float, default=0.20, help="Maximum crisis ratio (0–1)")
    args = parser.parse_args()

    # Load or synthesize conversations
    conversations: List[Conversation]
    if args.input:
        conversations = read_conversations_jsonl(Path(args.input))
    else:
        # Synthesize a larger set of unique conversations by cloning with new IDs
        base = build_sample_conversations()
        conversations = []
        # Aim for roughly 2x target to allow balancing
        multiplier = max((args.target_count * 2) // max(1, len(base)), 1)
        for i in range(multiplier):
            for b in base:
                if hasattr(b, "to_dict") and hasattr(type(b), "from_dict"):
                    data = b.to_dict()
                    data["conversation_id"] = str(uuid.uuid4())
                    conversations.append(Conversation.from_dict(data))
                else:
                    # Fallback: create a minimal copy
                    data = {
                        "conversation_id": str(uuid.uuid4()),
                        "source": b.source,
                        "messages": [
                            {"role": m.role, "content": m.content, "timestamp": m.timestamp, "metadata": m.metadata}
                            for m in b.messages
                        ],
                        "metadata": getattr(b, "metadata", {}),
                        "created_at": getattr(b, "created_at", None),
                        "updated_at": getattr(b, "updated_at", None),
                    }
                    conversations.append(Conversation.from_dict(data))

    # Curate and balance
    curated = ExpertValidationDataset.curate_from_conversations(conversations)
    curated = balance_examples(curated, target=args.target_count, min_crisis_ratio=args.min_crisis, max_crisis_ratio=args.max_crisis)

    # Build dataset and export
    dataset = ExpertValidationDataset(dataset_id=str(uuid.uuid4()), examples=curated)
    out_path = Path(args.output)
    dataset.to_jsonl(out_path)

    # Create and save training manifest referencing the dataset
    manifest = create_safety_aware_manifest(str(out_path), dataset_version="1.0")
    manifest.dataset.name = "pixel_expert_validation"
    manifest.dataset.record_count = len(curated)
    manifest.dataset.created_at = manifest.created_at
    # Calculate checksum/size if file exists
    if out_path.exists():
        manifest.dataset.size_bytes = out_path.stat().st_size
        # Use checksum utility
        try:
            manifest.dataset.checksum = manifest.dataset.calculate_checksum(str(out_path))
        except Exception:
            pass

    # Save manifest next to dataset
    manifest_path = out_path.with_suffix("")
    manifest_path = manifest_path.parent / (manifest_path.name + "_training_manifest.json")
    manifest.save_to_file(str(manifest_path))

    print(json.dumps({
        "output": str(out_path),
        "count": len(curated),
        "manifest": str(manifest_path),
    }))


if __name__ == "__main__":
    main()
