#!/usr/bin/env python3
"""
Stage-based organization for training data per MasterTrainingPlan.md.

Reorganizes flat dataset into stage-based structure:
ai/training_data_consolidated/final/
├── MASTER_STAGE_1.jsonl  # Foundation & Rapport (40%)
├── MASTER_STAGE_2.jsonl  # Therapeutic Expertise (25%)
├── MASTER_STAGE_3.jsonl  # Edge Stress Test (20%)
├── MASTER_STAGE_4.jsonl  # Voice/Persona (15%)
└── MASTER_STAGE_5.jsonl  # Safety/DPO (5%)

Usage:
    python stage_organizer.py <input_path> <output_dir>
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, TextIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Stage configuration per MasterTrainingPlan.md
STAGE_CONFIGS = {
    "stage1_foundation": {
        "target_percentage": 0.40,
        "quality_profile": {
            "empathy_min": 0.55,
            "safety_min": 0.75,
        },
        "file_name": "MASTER_STAGE_1.jsonl",
    },
    "stage2_therapeutic_expertise": {
        "target_percentage": 0.25,
        "quality_profile": {
            "empathy_min": 0.50,
            "reasoning_score_min": 0.65,
        },
        "file_name": "MASTER_STAGE_2.jsonl",
    },
    "stage3_edge_stress_test": {
        "target_percentage": 0.20,
        "quality_profile": {
            "crisis_intensity_required": True,
            "is_training_edge_case": True,
        },
        "file_name": "MASTER_STAGE_3.jsonl",
    },
    "stage4_voice_persona": {
        "target_percentage": 0.15,
        "quality_profile": {
            "voice_signature_required": True,
            "empathy_min": 0.60,
        },
        "file_name": "MASTER_STAGE_4.jsonl",
    },
    "stage5_safety_dpo": {
        "target_percentage": 0.05,
        "quality_profile": {
            "dpo_preference_required": True,
        },
        "file_name": "MASTER_STAGE_5.jsonl",
    },
}


def classify_stage(conversation: dict) -> str:
    """
    Classify conversation into stage based on metadata.

    Priority order (highest to lowest):
    - stage4_voice_persona: Has voice_signature or persona_id
    - stage3_edge_stress_test: Has is_training_edge_case
    - stage2_therapeutic_expertise: Has chain_of_thought or reasoning_score
    - stage1_foundation: Has empathy_score (therapeutic content)
    - supplementary: Default for unclassifiable

    Returns:
        Stage name or 'supplementary' if unclassifiable
    """
    metadata = conversation.get("metadata", {})

    # Stage 4: Voice/Persona (highest priority)
    if "voice_signature" in metadata or "persona_id" in metadata:
        return "stage4_voice_persona"

    # Stage 3: Edge cases
    if metadata.get("is_training_edge_case"):
        return "stage3_edge_stress_test"

    # Stage 2: Reasoning/CoT
    if "chain_of_thought" in metadata or "reasoning_score" in metadata:
        return "stage2_therapeutic_expertise"

    # Stage 1: Foundation (therapeutic content)
    if "empathy_score" in metadata or "therapeutic_content" in metadata:
        return "stage1_foundation"

    # Stage 5: Safety/DPO (only if explicitly marked)
    if metadata.get("dpo_preference_required") or metadata.get("safety_dpo"):
        return "stage5_safety_dpo"

    return "supplementary"


def organize_by_stage(input_path: Path, output_dir: Path) -> Dict[str, int]:
    """
    Read flat dataset and organize into stage-based files.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory for output stage files

    Returns:
        Count per stage
    """
    logger.info(f"Reading input file: {input_path}")

    stage_counts: Dict[str, int] = defaultdict(int)
    stage_counts["supplementary"] = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    # Open output files
    output_handles: Dict[str, TextIO] = {}
    stage_files: Dict[str, Path] = {}

    for stage, config in STAGE_CONFIGS.items():
        file_path = output_dir / config["file_name"]
        stage_files[stage] = file_path
        output_handles[stage] = open(file_path, "w")

    # Handle supplementary
    supplementary_path = output_dir / "SUPPLEMENTARY.jsonl"
    stage_files["supplementary"] = supplementary_path
    output_handles["supplementary"] = open(supplementary_path, "w")

    total_count = 0

    try:
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                conversation = json.loads(line)
                stage = classify_stage(conversation)
                stage_counts[stage] += 1
                total_count += 1

                # Write to appropriate stage file
                json.dump(conversation, output_handles[stage])
                output_handles[stage].write("\n")

                # Progress every 10000
                if total_count % 10000 == 0:
                    logger.info(f"Processed {total_count} conversations...")

    finally:
        for handle in output_handles.values():
            handle.close()

    # Log results
    logger.info(f"Organization complete. Total: {total_count}")
    for stage, count in sorted(stage_counts.items()):
        if count > 0:
            percentage = (count / total_count * 100) if total_count > 0 else 0
            logger.info(f"  {stage}: {count} ({percentage:.1f}%)")

    return dict(stage_counts)


def create_manifest(output_dir: Path, stage_counts: Dict[str, int]) -> dict:
    """Create manifest file with stage metadata."""
    manifest = {
        "stages": {},
        "total_conversations": sum(stage_counts.values()),
        "last_updated": Path(output_dir).stat().st_mtime,
    }

    for stage, config in STAGE_CONFIGS.items():
        count = stage_counts.get(stage, 0)
        manifest["stages"][stage] = {
            "target_percentage": config["target_percentage"],
            "actual_count": count,
            "actual_percentage": count / manifest["total_conversations"] if manifest["total_conversations"] > 0 else 0,
            "quality_profile": config["quality_profile"],
            "file_path": f"{output_dir}/{config['file_name']}",
        }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Manifest written to: {manifest_path}")
    return manifest


def main():
    if len(sys.argv) != 3:
        print("Usage: python stage_organizer.py <input_path> <output_dir>")
        print("Example: python stage_organizer.py data/flat.jsonl data/stages/")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    stage_counts = organize_by_stage(input_path, output_dir)
    create_manifest(output_dir, stage_counts)

    logger.info("Stage organization complete!")


if __name__ == "__main__":
    main()
