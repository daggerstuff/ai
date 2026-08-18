#!/usr/bin/env python3
"""
S3-Native Streaming Stage Organizer - reads from S3, writes to gdrive (which has write access).

Uses the original S3Streamer for reading, rclone for writing to gdrive.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset_pipeline.extractors.s3_streamer import S3Streamer

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

    def split_filename(self, split_name: str) -> str:
        stage_num = int(self.stage.value.split("_")[0].replace("stage", ""))
        return f"MASTER_STAGE_{stage_num}_{split_name}.jsonl"


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


def stream_s3_shards(streamer: S3Streamer, prefix: str = "compiled_dataset/") -> Iterator[dict]:
    """Stream all JSONL records from S3 shards under prefix."""
    for key in streamer.list_files(prefix):
        if key.endswith(".jsonl"):
            logger.info(f"Streaming: {key}")
            try:
                yield from streamer.stream_jsonl(key)
            except Exception as e:
                logger.warning(f"Failed to stream {key}: {e}")


class GDriveWriter:
    """Writes JSONL records to gdrive using rclone rcat."""

    def __init__(
        self,
        gdrive_remote: str = "gdrive:",
        gdrive_base: str = "pixeldata/final_dataset/v5_stages/",
        chunk_size: int = 50000,
    ):
        self.gdrive_remote = gdrive_remote
        self.gdrive_base = gdrive_base
        self.chunk_size = chunk_size
        self._procs: dict[str, subprocess.Popen] = {}
        self._counts: dict[str, int] = defaultdict(int)

    def _get_proc(self, filename: str) -> subprocess.Popen:
        """Get or create rclone rcat process for a file."""
        if filename not in self._procs:
            cmd = ["rclone", "rcat", f"{self.gdrive_remote}{self.gdrive_base}{filename}"]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, text=True, encoding="utf-8")
            assert proc.stdin is not None
            self._procs[filename] = proc
            self._counts[filename] = 0
        return self._procs[filename]

    def write(self, filename: str, record: dict[str, Any]) -> None:
        """Write a single record."""
        proc = self._get_proc(filename)
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(record) + "\n")
        self._counts[filename] += 1

        if self._counts[filename] % self.chunk_size == 0:
            proc.stdin.flush()
            logger.debug(f"Flushed {self._counts[filename]} records to {filename}")

    def close_all(self) -> None:
        """Close all processes and verify success."""
        for filename, proc in self._procs.items():
            if proc.stdin:
                proc.stdin.close()
            return_code = proc.wait()
            count = self._counts[filename]
            if return_code != 0:
                raise Exception(f"rclone rcat failed for {filename} with return code {return_code}")
            logger.info(f"Successfully uploaded {count} records to gdrive:{self.gdrive_base}{filename}")
        self._procs.clear()
        self._counts.clear()


class S3StageOrganizer:
    def __init__(
        self,
        s3_prefix: str = "compiled_dataset/",
        gdrive_remote: str = "gdrive:",
        gdrive_base: str = "pixeldata/final_dataset/v5_stages/",
        configs: dict[Stage, StageConfig] | None = None,
        seed: int = 42,
        chunk_size: int = 50000,
    ) -> None:
        self.streamer = S3Streamer()
        self.s3_prefix = s3_prefix
        self.gdrive_remote = gdrive_remote
        self.gdrive_base = gdrive_base
        self.configs = configs or DEFAULT_STAGE_CONFIGS
        self.seed = seed
        self.chunk_size = chunk_size

    def count_stages(self) -> tuple[dict[Stage, int], int]:
        """Pass 1: Count records per stage by streaming from S3."""
        stage_counts: dict[Stage, int] = defaultdict(int)
        total = 0

        logger.info(f"Pass 1: Counting stages from s3://pixeldata/{self.s3_prefix}")

        for record in stream_s3_shards(self.streamer, self.s3_prefix):
            stage = classify_record(record)
            stage_counts[stage] += 1
            total += 1

            if total % 100000 == 0:
                logger.info(f"  Counted {total} records...")

        logger.info(f"Pass 1 complete: {total} total records")
        for stage, count in stage_counts.items():
            logger.info(f"  {stage.value}: {count}")

        return dict(stage_counts), total

    def _make_stage_generator(
        self,
        target_stage: Stage,
        target_count: int,
        rng: random.Random,
    ) -> Iterator[dict]:
        """Create a generator that yields records for a specific stage with quota enforcement."""
        yielded = 0
        for record in stream_s3_shards(self.streamer, self.s3_prefix):
            if yielded >= target_count:
                break
            stage = classify_record(record)
            if stage == target_stage:
                yielded += 1
                yield record
        logger.info(f"Generator for {target_stage.value} yielded {yielded} records")

    def _make_split_generator(
        self,
        target_stage: Stage,
        target_count: int,
        split_name: str,
        rng: random.Random,
    ) -> Iterator[dict]:
        """Create a generator that yields records for a specific stage AND split."""
        yielded = 0
        split_thresholds = {"train": 0.80, "val": 0.90, "test": 1.0}
        threshold = split_thresholds[split_name]
        prev_threshold = {"train": 0.0, "val": 0.80, "test": 0.90}[split_name]

        for record in stream_s3_shards(self.streamer, self.s3_prefix):
            if yielded >= target_count:
                break
            stage = classify_record(record)
            if stage != target_stage:
                continue

            r = rng.random()
            if prev_threshold <= r < threshold:
                yielded += 1
                yield record

        logger.info(f"Generator for {target_stage.value}/{split_name} yielded {yielded} records")

    def write_stages(self, stage_counts: dict[Stage, int], total: int) -> dict[Stage, dict[str, int]]:
        """Pass 2: Use GDriveWriter with generators for each stage/split."""
        target_counts = {}
        for stage, config in self.configs.items():
            target_counts[stage] = int(total * config.target_percentage)

        rng = random.Random(self.seed)

        # Calculate split target counts
        split_targets: dict[Stage, dict[str, int]] = {}
        for stage in Stage:
            tc = target_counts[stage]
            split_targets[stage] = {
                "train": int(tc * 0.80),
                "val": int(tc * 0.10),
                "test": tc - int(tc * 0.80) - int(tc * 0.10),
            }

        actual_counts: dict[Stage, dict[str, int]] = {stage: {"train": 0, "val": 0, "test": 0} for stage in Stage}

        logger.info("Pass 2: Writing stage files to gdrive via rcat streaming")

        # Create writers for each output file
        manifest_writers = {
            stage: GDriveWriter(self.gdrive_remote, self.gdrive_base, self.chunk_size) for stage in Stage
        }
        split_writers = {
            (stage, split): GDriveWriter(self.gdrive_remote, self.gdrive_base, self.chunk_size)
            for stage in Stage
            for split in ["train", "val", "test"]
        }

        # For each stage, stream and write
        for stage in Stage:
            config = self.configs[stage]
            target = target_counts[stage]

            if target == 0:
                logger.info(f"  {stage.value}: target 0, skipping")
                continue

            # Write main manifest
            logger.info(f"  Writing manifest for {stage.value} (target: {target})")
            for record in self._make_stage_generator(stage, target, rng):
                manifest_writers[stage].write(config.manifest_filename, record)

            # Write splits
            for split_name in ["train", "val", "test"]:
                split_target = split_targets[stage][split_name]
                if split_target == 0:
                    continue

                logger.info(f"  Writing {split_name} split for {stage.value} (target: {split_target})")
                for record in self._make_split_generator(stage, split_target, split_name, rng):
                    split_writers[(stage, split_name)].write(config.split_filename(split_name), record)

        # Close all writers (flushes and verifies)
        logger.info("Closing writers and verifying uploads...")
        for stage in Stage:
            manifest_writers[stage].close_all()
            for split_name in ["train", "val", "test"]:
                split_writers[(stage, split_name)].close_all()

        # Verify actual counts by re-streaming from gdrive
        logger.info("Verifying written counts...")
        for stage in Stage:
            config = self.configs[stage]
            for split_name in ["train", "val", "test"]:
                filename = config.split_filename(split_name)
                try:
                    # Use rclone cat to count
                    result = subprocess.run(
                        ["rclone", "cat", f"{self.gdrive_remote}{self.gdrive_base}{filename}"],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    count = sum(1 for _ in result.stdout.split("\n") if _.strip())
                    actual_counts[stage][split_name] = count
                except Exception as e:
                    logger.warning(f"Could not verify {filename}: {e}")
                    actual_counts[stage][split_name] = 0

        logger.info("Pass 2 complete")
        for stage in Stage:
            config = self.configs[stage]
            written = sum(actual_counts[stage].values())
            target = target_counts[stage]
            if written > 0:
                logger.info(
                    f"  {stage.value}: {written} written (target {target}) "
                    f"-> train={actual_counts[stage]['train']}, "
                    f"val={actual_counts[stage]['val']}, "
                    f"test={actual_counts[stage]['test']}"
                )

        return actual_counts

    def write_manifests_index(self, actual_counts: dict[Stage, dict[str, int]]) -> None:
        """Write manifests.json index file to gdrive."""
        manifests = []
        for stage, config in self.configs.items():
            actual = sum(actual_counts[stage].values())
            if actual > 0:
                manifests.append(
                    {
                        "stage": stage.value,
                        "target_percentage": config.target_percentage,
                        "actual_count": actual,
                        "quality_profile": config.quality_profile,
                        "split_counts": actual_counts[stage],
                        "manifest_filename": config.manifest_filename,
                        "gdrive_path": f"gdrive:{self.gdrive_base}{config.manifest_filename}",
                    }
                )

        # Write to local temp file then rclone copy
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifests, f, indent=2)
            temp_path = f.name

        subprocess.run(
            ["rclone", "copy", temp_path, f"{self.gdrive_remote}{self.gdrive_base}manifests.json"], check=True
        )
        logger.info(f"Wrote manifests index to gdrive:{self.gdrive_base}manifests.json")

    def organize(self) -> None:
        """Run the full two-pass organization: read S3, write gdrive."""
        # Pass 1: Count
        stage_counts, total = self.count_stages()

        if total == 0:
            logger.error("No records found in S3")
            return

        # Pass 2: Write with quotas using streaming generators
        actual_counts = self.write_stages(stage_counts, total)

        # Write index
        self.write_manifests_index(actual_counts)

        logger.info(f"Organization complete: {total} records organized")


def main() -> None:
    parser = argparse.ArgumentParser(description="S3->gdrive streaming stage organizer")
    parser.add_argument("--s3-prefix", type=str, default="compiled_dataset/", help="S3 prefix to read from")
    parser.add_argument("--gdrive-remote", type=str, default="gdrive:", help="gdrive remote name")
    parser.add_argument(
        "--gdrive-base", type=str, default="pixeldata/final_dataset/v5_stages/", help="gdrive base path"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Records per flush")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    organizer = S3StageOrganizer(
        s3_prefix=args.s3_prefix,
        gdrive_remote=args.gdrive_remote,
        gdrive_base=args.gdrive_base,
        seed=args.seed,
        chunk_size=args.chunk_size,
    )
    organizer.organize()


if __name__ == "__main__":
    main()
