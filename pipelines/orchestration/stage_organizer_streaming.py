#!/usr/bin/env python3
"""
Streaming Stage Organizer - processes large datasets without loading all into memory.

Two-pass approach:
1. First pass: stream through all shards, count records per stage (minimal memory)
2. Second pass: stream again, write records to stage files with quota enforcement
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    STAGE1_FOUNDATION = "stage1_foundation"
    STAGE2_THERAPEUTIC_EXPERTISE = "stage2_therapeutic_expertise"
    STAGE3_EDGE_STRESS_TEST = "stage3_edge_stress_test"
    STAGE4_VOICE_PERSONA = "stage4_voice_persona"
    STAGE5_SAFETY = "stage5_safety"


_SAFETY_SOURCES: frozenset[str] = frozenset(
    {
        "safety",
        "crisis_intervention",
        "harm_prevention",
        "suicide_prevention",
        "self_harm",
        "crisis_hotline",
    }
)

_SAFETY_TAGS: frozenset[str] = frozenset(
    {
        "safety",
        "crisis",
        "self_harm",
        "suicide",
        "harm_prevention",
        "emergency",
        "crisis_response",
        "mental_health_crisis",
    }
)

_VOICE_PERSONA_SOURCES: frozenset[str] = frozenset(
    {
        "pixel_voice",
        "voice_persona",
        "dual_persona",
        "persona",
    }
)

_VOICE_PERSONA_TAGS: frozenset[str] = frozenset(
    {
        "voice",
        "persona",
        "dual_persona",
        "personality",
        "character",
        "role_play",
    }
)

_EDGE_CASE_SOURCES: frozenset[str] = frozenset(
    {
        "edge_cases",
        "adversarial",
        "stress_test",
        "jailbreak",
        "red_team",
        "safety_test",
    }
)

_EDGE_CASE_TAGS: frozenset[str] = frozenset(
    {
        "edge_case",
        "adversarial",
        "stress_test",
        "jailbreak",
        "red_team",
        "crisis",
        "boundary_test",
    }
)

_THERAPEUTIC_MODALITIES: frozenset[str] = frozenset(
    {
        "cbt",
        "dbt",
        "psychodynamic",
        "emdr",
        "act",
        "mbct",
        "ipt",
        "sfbt",
        "gottman",
        "somatic",
        "trauma_informed",
        "attachment_based",
    }
)

_THERAPEUTIC_TAGS: frozenset[str] = frozenset(
    {
        "therapy",
        "counseling",
        "psychotherapy",
        "clinical",
        "diagnosis",
        "treatment_plan",
        "intervention",
        "therapeutic_technique",
        "case_study",
    }
)

_FOUNDATION_TAGS: frozenset[str] = frozenset(
    {
        "psychology",
        "mental_health",
        "general",
        "education",
        "self_help",
        "wellness",
        "mindfulness",
        "communication",
        "emotional_intelligence",
    }
)


@dataclass(frozen=True)
class StageConfig:
    stage: Stage
    target_percentage: float
    quality_profile: dict[str, float] = field(default_factory=dict)

    @property
    def manifest_filename(self) -> str:
        stage_num = int(self.stage.value.split("_")[0].replace("stage", ""))
        return f"MASTER_STAGE_{stage_num}.jsonl"


DEFAULT_STAGE_CONFIGS: dict[Stage, StageConfig] = {
    Stage.STAGE1_FOUNDATION: StageConfig(
        stage=Stage.STAGE1_FOUNDATION,
        target_percentage=0.35,
        quality_profile={"empathy_floor": 0.70, "clinical_floor": 0.30, "safety_floor": 1.0},
    ),
    Stage.STAGE2_THERAPEUTIC_EXPERTISE: StageConfig(
        stage=Stage.STAGE2_THERAPEUTIC_EXPERTISE,
        target_percentage=0.25,
        quality_profile={"empathy_floor": 0.75, "clinical_floor": 0.50, "safety_floor": 1.0},
    ),
    Stage.STAGE3_EDGE_STRESS_TEST: StageConfig(
        stage=Stage.STAGE3_EDGE_STRESS_TEST,
        target_percentage=0.20,
        quality_profile={"empathy_floor": 0.60, "clinical_floor": 0.40, "safety_floor": 1.0},
    ),
    Stage.STAGE4_VOICE_PERSONA: StageConfig(
        stage=Stage.STAGE4_VOICE_PERSONA,
        target_percentage=0.15,
        quality_profile={"empathy_floor": 0.80, "clinical_floor": 0.35, "safety_floor": 1.0},
    ),
    Stage.STAGE5_SAFETY: StageConfig(
        stage=Stage.STAGE5_SAFETY,
        target_percentage=0.05,
        quality_profile={"empathy_floor": 0.65, "clinical_floor": 0.45, "safety_floor": 1.0},
    ),
}


def classify_record(record: dict[str, Any]) -> Stage:
    source = (record.get("source") or "").lower()
    metadata = record.get("metadata", {}) or {}
    topic_tags = [t.lower() for t in (metadata.get("topic_tags") or [])]
    therapeutic_modality = (metadata.get("therapeutic_modality") or "").lower()

    if source in _SAFETY_SOURCES or any(tag in _SAFETY_TAGS for tag in topic_tags):
        return Stage.STAGE5_SAFETY

    if source in _VOICE_PERSONA_SOURCES or any(tag in _VOICE_PERSONA_TAGS for tag in topic_tags):
        return Stage.STAGE4_VOICE_PERSONA

    if source in _EDGE_CASE_SOURCES or any(tag in _EDGE_CASE_TAGS for tag in topic_tags):
        return Stage.STAGE3_EDGE_STRESS_TEST

    if therapeutic_modality in _THERAPEUTIC_MODALITIES or any(tag in _THERAPEUTIC_TAGS for tag in topic_tags):
        return Stage.STAGE2_THERAPEUTIC_EXPERTISE

    if any(tag in _FOUNDATION_TAGS for tag in topic_tags):
        return Stage.STAGE1_FOUNDATION

    return Stage.STAGE1_FOUNDATION


class StreamingStageOrganizer:
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        configs: dict[Stage, StageConfig] | None = None,
        seed: int = 42,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.configs = configs or DEFAULT_STAGE_CONFIGS
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def count_stages(self) -> tuple[dict[Stage, int], int]:
        """Pass 1: Count records per stage."""
        stage_counts: dict[Stage, int] = defaultdict(int)
        total = 0

        shard_files = sorted(self.input_dir.glob("*.jsonl"))
        logger.info(f"Pass 1: Counting stages across {len(shard_files)} shards")

        for shard_path in shard_files:
            logger.info(f"  Counting: {shard_path.name}")
            with open(shard_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        stage = classify_record(record)
                        stage_counts[stage] += 1
                        total += 1
                    except json.JSONDecodeError:
                        continue

        logger.info(f"Pass 1 complete: {total} total records")
        for stage, count in stage_counts.items():
            logger.info(f"  {stage.value}: {count}")

        return dict(stage_counts), total

    def write_stages(self, stage_counts: dict[Stage, int], total: int) -> dict[Stage, dict[str, int]]:
        """Pass 2: Stream records and write to stage files with quota enforcement."""
        # Calculate target counts per stage
        target_counts = {}
        for stage, config in self.configs.items():
            target_counts[stage] = int(total * config.target_percentage)

        # Track written counts per stage
        written_counts: dict[Stage, int] = defaultdict(int)
        split_counts: dict[Stage, dict[str, int]] = {stage: {"train": 0, "val": 0, "test": 0} for stage in Stage}

        # Open output files for each stage
        stage_files = {}
        stage_split_files = {}
        for stage, config in self.configs.items():
            # Main manifest file
            manifest_path = self.output_dir / config.manifest_filename
            stage_files[stage] = open(manifest_path, "w")

            # Split files
            for split_name in ["train", "val", "test"]:
                split_path = self.output_dir / f"{config.manifest_filename.replace('.jsonl', f'_{split_name}.jsonl')}"
                stage_split_files[(stage, split_name)] = open(split_path, "w")

        # Random generators for splitting
        rng = random.Random(self.seed)

        shard_files = sorted(self.input_dir.glob("*.jsonl"))
        logger.info(f"Pass 2: Writing stage files across {len(shard_files)} shards")

        for shard_path in shard_files:
            logger.info(f"  Writing: {shard_path.name}")
            with open(shard_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        stage = classify_record(record)

                        # Check quota
                        if written_counts[stage] >= target_counts[stage]:
                            continue

                        # Write to main manifest
                        stage_files[stage].write(line + "\n")

                        # Determine split (80/10/10)
                        r = rng.random()
                        if r < 0.80:
                            split_name = "train"
                        elif r < 0.90:
                            split_name = "val"
                        else:
                            split_name = "test"

                        # Write to split file
                        stage_split_files[(stage, split_name)].write(line + "\n")

                        written_counts[stage] += 1
                        split_counts[stage][split_name] += 1

                    except json.JSONDecodeError:
                        continue

        # Close all files
        for f in stage_files.values():
            f.close()
        for f in stage_split_files.values():
            f.close()

        logger.info("Pass 2 complete")
        for stage in Stage:
            config = self.configs[stage]
            logger.info(
                f"  {stage.value}: {written_counts[stage]} written "
                f"(target {target_counts[stage]}) "
                f"-> train={split_counts[stage]['train']}, "
                f"val={split_counts[stage]['val']}, "
                f"test={split_counts[stage]['test']}"
            )

        return dict(split_counts)

    def write_manifests_index(self, split_counts: dict[Stage, dict[str, int]]) -> None:
        """Write manifests.json index file."""
        manifests = []
        for stage, config in self.configs.items():
            if split_counts[stage]["train"] + split_counts[stage]["val"] + split_counts[stage]["test"] > 0:
                manifests.append(
                    {
                        "stage": stage.value,
                        "target_percentage": config.target_percentage,
                        "actual_count": sum(split_counts[stage].values()),
                        "quality_profile": config.quality_profile,
                        "split_counts": split_counts[stage],
                        "manifest_file": config.manifest_filename,
                        "output_path": str(self.output_dir / config.manifest_filename),
                    }
                )

        index_path = self.output_dir / "manifests.json"
        with open(index_path, "w") as f:
            json.dump(manifests, f, indent=2)
        logger.info(f"Wrote manifests index to {index_path}")

    def organize(self) -> None:
        """Run the full two-pass organization."""
        # Pass 1: Count
        stage_counts, total = self.count_stages()

        # Pass 2: Write with quotas
        split_counts = self.write_stages(stage_counts, total)

        # Write index
        self.write_manifests_index(split_counts)

        logger.info(
            f"Organization complete: {total} records organized into {len([s for s in split_counts if sum(split_counts[s].values()) > 0])} stages"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming stage organizer for large datasets")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing JSONL shard files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("ai/training_data_consolidated/final"), help="Output directory"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    organizer = StreamingStageOrganizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    organizer.organize()


if __name__ == "__main__":
    main()
