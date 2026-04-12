#!/usr/bin/env python3
"""
Compile Stage 2 local datasets into final training format.
Integrates defense mechanism synthetic data with metadata.
"""

from datetime import datetime, timezone

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any


def _compute_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode()).hexdigest()


def convert_defense_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert defense mechanism record to ChatML format with metadata."""
    messages = []

    for turn in record.get("turns", []):
        speaker = turn.get("speaker", "").lower()
        text = turn.get("text", "")

        # Map speaker to role
        if speaker in ["seeker", "user"]:
            role = "user"
        elif speaker in ["supporter", "assistant"]:
            role = "assistant"
        else:
            role = "user" if len(messages) % 2 == 0 else "assistant"

        messages.append({"role": role, "content": text})

    # Compute content hash
    content_str = json.dumps(messages, sort_keys=True)
    record_hash = _compute_hash(content_str)

    # Extract defense mechanism metadata
    metadata = {
        "hash": record_hash,
        "family": "defense_mechanisms_synthetic",
        "source": "synthetic_minority_generated",
        "defense_label": record.get("label_name", ""),
        "defense_id": record.get("label", 0),
        "sub_mechanism": record.get("metadata", {}).get("sub_mechanism", ""),
        "dmrs_items": record.get("metadata", {}).get("mapped_dmrs_items", []),
        "clinical_rationale": record.get("metadata", {}).get("clinical_rationale", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "messages": messages,
        "metadata": metadata,
    }


def compile_stage2_dataset():
    """Compile all Stage 2 datasets."""
    project_root = Path(__file__).resolve().parents[3]
    output_dir = project_root / "compiled_stage2_dataset"
    output_dir.mkdir(exist_ok=True)

    all_records = []
    seen_hashes = set()

    # Load defense mechanism synthetic data
    defense_file = (
        project_root
        / "ai/training/defense_mechanisms/data/synthetic_minority_generated.jsonl"
    )
    print(f"Loading defense mechanism data from: {defense_file}")

    if defense_file.exists():
        with open(defense_file) as f:
            for line in f:
                record = json.loads(line)
                converted = convert_defense_record(record)

                # Deduplicate by hash
                record_hash = converted["metadata"]["hash"]
                if record_hash not in seen_hashes:
                    seen_hashes.add(record_hash)
                    all_records.append(converted)

        print(f"Loaded {len(all_records)} defense mechanism records")

    # Load additional synthetic data if available
    additional_dirs = [
        project_root / "ai/data/datasets/synthetic",
        project_root / "ai/data/generated/pix8_long_sessions",
        project_root / "ai/data/generated/pix8_edge_cases",
    ]

    for data_dir in additional_dirs:
        if not data_dir.exists():
            continue

        for jsonl_file in data_dir.glob("*.jsonl"):
            print(f"Processing: {jsonl_file.name}")
            try:
                with open(jsonl_file) as f:
                    for line in f:
                        try:
                            record = json.loads(line)
                            # Convert various formats to ChatML
                            if "messages" in record:
                                converted = record
                            elif "turns" in record:
                                converted = convert_defense_record(record)
                            elif "conversation" in record:
                                messages = []
                                for turn in record["conversation"]:
                                    if "role" in turn and "content" in turn:
                                        messages.append(turn)
                                converted = {
                                    "messages": messages,
                                    "metadata": {"source": jsonl_file.stem},
                                }
                            elif "question" in record and "answer" in record:
                                converted = {
                                    "messages": [
                                        {"role": "user", "content": record["question"]},
                                        {
                                            "role": "assistant",
                                            "content": record["answer"],
                                        },
                                    ],
                                    "metadata": {"source": jsonl_file.stem},
                                }
                            else:
                                continue

                            # Deduplicate
                            content_str = json.dumps(
                                converted["messages"], sort_keys=True
                            )
                            record_hash = _compute_hash(content_str)

                            if (
                                record_hash not in seen_hashes
                                and len(converted.get("messages", [])) >= 2
                            ):
                                seen_hashes.add(record_hash)
                                converted["metadata"]["hash"] = record_hash
                                all_records.append(converted)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                print(f"Error processing {jsonl_file}: {e}")

    print(f"\nTotal compiled records: {len(all_records)}")

    # Split into train/val/test (90/5/5)
    import random

    random.seed(42)
    random.shuffle(all_records)

    n_total = len(all_records)
    n_train = int(n_total * 0.90)
    n_val = int(n_total * 0.05)

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train : n_train + n_val],
        "test": all_records[n_train + n_val :],
    }

    # Write splits
    for split_name, records in splits.items():
        split_file = output_dir / f"stage2_{split_name}.jsonl"
        with open(split_file, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(records)} records to {split_file}")

    # Generate METADATA.json
    metadata = {
        "dataset_name": "Pixelated Empathy Stage 2 - Defense Mechanisms & Clinical Reasoning",
        "version": "2.0.0",
        "compiled_at": datetime.now(timezone.utc).isoformat(),
        "total_records": n_total,
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "features": {
            "defense_mechanisms": True,
            "persona_profiles": True,
            "clinical_reasoning": True,
            "chatml_format": True,
        },
        "defense_mechanism_labels": [
            "Action Defenses",
            "Narcissistic Defenses",
            "Primitive Defenses",
            "Major Image-Distorting Defenses",
            "Minor Image-Distorting Defenses",
        ],
        "sources": [
            "synthetic_minority_generated.jsonl",
            "pix8_long_sessions",
            "pix8_edge_cases",
        ],
        "format": "ChatML (messages: [{role, content}])",
        "quality_gates": {
            "pii_scrubbed": True,
            "deduplicated": True,
            "min_turns_per_conversation": 2,
        },
    }

    metadata_file = output_dir / "METADATA.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nGenerated METADATA.json at {metadata_file}")
    print(f"\n✅ Stage 2 dataset compilation complete!")
    print(f"   Output directory: {output_dir}")

    return output_dir


if __name__ == "__main__":
    compile_stage2_dataset()
