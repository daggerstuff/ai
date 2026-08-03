"""
PIX-4193: Stage-Based Dataset Directory Organizer

Organizes compiled dataset shards into stage-specific manifests with
strict quota management and 80/10/10 train/val/test splits.

Usage:
    uv run python -m ai.pipelines.orchestration.stage_organizer \
        --input-dir ai/data/compiled_dataset \
        --output-dir ai/training_data_consolidated/final
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Stage(StrEnum):
    """Training stage identifiers with target percentages."""

    STAGE1_FOUNDATION = "stage1_foundation"
    STAGE2_THERAPEUTIC_EXPERTISE = "stage2_therapeutic_expertise"
    STAGE3_EDGE_STRESS_TEST = "stage3_edge_stress_test"
    STAGE4_VOICE_PERSONA = "stage4_voice_persona"
    STAGE5_SAFETY = "stage5_safety"


# Stage 5 safety indicators (new for PIX-4193)
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

# Stage 4 voice/persona indicators
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

# Stage 3 edge/stress test indicators
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

# Stage 2 therapeutic expertise indicators
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

# Stage 1 foundation indicators
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
    """Configuration for a single training stage."""

    stage: Stage
    target_percentage: float
    quality_profile: dict[str, float] = field(default_factory=dict)

    @property
    def manifest_filename(self) -> str:
        """Generate manifest filename like MASTER_STAGE_1.jsonl."""
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


@dataclass
class StageManifest:
    """Manifest for a single stage's dataset."""

    stage: str
    target_percentage: float
    actual_count: int
    quality_profile: dict[str, float]
    split_counts: dict[str, int]  # {"train": N, "val": N, "test": N}
    manifest_file: str
    output_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "target_percentage": self.target_percentage,
            "actual_count": self.actual_count,
            "quality_profile": self.quality_profile,
            "split_counts": self.split_counts,
            "manifest_file": self.manifest_file,
            "output_path": self.output_path,
        }


def classify_record(record: dict[str, Any]) -> Stage:
    """
    Classify a single record into a training stage.

    Priority order (highest to lowest):
      1. Stage 5 — Safety/crisis content
      2. Stage 4 — Voice/persona content
      3. Stage 3 — Edge cases and stress tests
      4. Stage 2 — Therapeutic expertise
      5. Stage 1 — Foundation content
    """
    source = (record.get("source") or "").lower()
    metadata = record.get("metadata", {}) or {}
    topic_tags = [t.lower() for t in (metadata.get("topic_tags") or [])]
    therapeutic_modality = (metadata.get("therapeutic_modality") or "").lower()

    # Check Stage 5: Safety
    if source in _SAFETY_SOURCES or any(tag in _SAFETY_TAGS for tag in topic_tags):
        return Stage.STAGE5_SAFETY

    # Check Stage 4: Voice/Persona
    if source in _VOICE_PERSONA_SOURCES or any(tag in _VOICE_PERSONA_TAGS for tag in topic_tags):
        return Stage.STAGE4_VOICE_PERSONA

    # Check Stage 3: Edge/Stress Test
    if source in _EDGE_CASE_SOURCES or any(tag in _EDGE_CASE_TAGS for tag in topic_tags):
        return Stage.STAGE3_EDGE_STRESS_TEST

    # Check Stage 2: Therapeutic Expertise
    if therapeutic_modality in _THERAPEUTIC_MODALITIES or any(tag in _THERAPEUTIC_TAGS for tag in topic_tags):
        return Stage.STAGE2_THERAPEUTIC_EXPERTISE

    # Check Stage 1: Foundation (default for psychology/mental health content)
    if any(tag in _FOUNDATION_TAGS for tag in topic_tags):
        return Stage.STAGE1_FOUNDATION

    # Default to Stage 1 for unclassified conversational content
    return Stage.STAGE1_FOUNDATION


def split_dataset(
    records: list[dict[str, Any]],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """
    Split records into train/val/test sets.

    Args:
        records: List of records to split
        train_ratio: Fraction for training (default 0.80)
        val_ratio: Fraction for validation (default 0.10)
        seed: Random seed for reproducibility

    Returns:
        Dict with "train", "val", "test" keys
    """
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def enforce_quotas(
    stage_records: dict[Stage, list[dict[str, Any]]],
    total_records: int,
    configs: dict[Stage, StageConfig],
) -> dict[Stage, list[dict[str, Any]]]:
    """
    Enforce stage quota percentages WITHOUT silent data loss (P0-4).

    Quotas are soft capacity targets, not hard caps. Under-represented
    stages keep all of their records; their unused capacity is redistributed
    to over-represented stages so overflow is absorbed instead of dropped.
    If overflow still exceeds the redistributable budget, records are
    retained anyway and a warning is logged - no record is ever silently
    discarded.

    Args:
        stage_records: Dict mapping stage to list of records
        total_records: Total number of records in dataset
        configs: Stage configuration dict

    Returns:
        Dict with record lists per stage; every input record is preserved
    """
    # Absolute target counts derived from configured percentages.
    targets: dict[Stage, int] = {
        stage: int(total_records * config.target_percentage) for stage, config in configs.items()
    }

    # Unused capacity from under-represented AND missing configured stages
    # is redistributable to over-target stages.
    remaining_slack = sum(
        max(0, targets.get(stage, 0) - len(records))
        for stage, records in stage_records.items()
        if stage in targets
    )
    # Missing configured stages contribute their full target as slack.
    for stage in configs:
        if stage not in stage_records:
            remaining_slack += targets[stage]

    result: dict[Stage, list[dict[str, Any]]] = {}

    for stage, records in stage_records.items():
        config = configs.get(stage)
        if not config:
            result[stage] = records
            continue

        target_count = targets[stage]
        if len(records) <= target_count:
            # Under (or at) target: keep everything.
            result[stage] = records
            continue

        # Over target: consume redistributable slack for this stage.
        overflow = len(records) - target_count
        if overflow > remaining_slack:
            logger.warning(
                f"Overflow for {stage.value} ({len(records)} records, overflow "
                f"{overflow}) exceeds remaining redistributable budget "
                f"({remaining_slack}); retaining all records to honor "
                f"no-silent-data-loss policy"
            )
        else:
            remaining_slack -= overflow
            logger.info(
                f"Redistributed quota for {stage.value}: retained {len(records)} "
                f"records (target {target_count}, overflow {overflow}, "
                f"remaining slack {remaining_slack})"
            )
        result[stage] = records

    return result


class StageOrganizer:
    """
    Organizes dataset shards into stage-specific manifests.

    Reads JSONL shards, classifies records into training stages,
    enforces quota percentages, splits 80/10/10, and writes manifests.
    """

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
        self._stage_records: dict[Stage, list[dict[str, Any]]] = {stage: [] for stage in Stage}

    def load_shards(self) -> int:
        """Load all JSONL shards from input directory."""
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        shard_files = sorted(self.input_dir.glob("*.jsonl"))
        if not shard_files:
            raise FileNotFoundError(f"No JSONL shards found in {self.input_dir}")

        total = 0
        for shard_path in shard_files:
            logger.info(f"Loading shard: {shard_path.name}")
            with open(shard_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        stage = classify_record(record)
                        self._stage_records[stage].append(record)
                        total += 1
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON in {shard_path.name}: {e}")

        logger.info(f"Loaded {total} records across {len(shard_files)} shards")
        return total

    def organize(self) -> list[StageManifest]:
        """
        Run the full organization pipeline.

        Returns:
            List of StageManifest objects for each stage
        """
        # Load and classify
        total = self.load_shards()

        # Enforce quotas
        self._stage_records = enforce_quotas(self._stage_records, total, self.configs)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        manifests: list[StageManifest] = []

        for stage, records in self._stage_records.items():
            if not records:
                logger.info(f"No records for {stage.value}, skipping")
                continue

            config = self.configs[stage]

            # Split 80/10/10
            splits = split_dataset(records, seed=self.seed)

            # Write split files
            for split_name, split_records in splits.items():
                if not split_records:
                    continue
                split_file = self.output_dir / f"{config.manifest_filename.replace('.jsonl', f'_{split_name}.jsonl')}"
                with open(split_file, "w") as f:
                    for record in split_records:
                        f.write(json.dumps(record) + "\n")

            # Write consolidated manifest file
            manifest_path = self.output_dir / config.manifest_filename
            with open(manifest_path, "w") as f:
                for record in records:
                    f.write(json.dumps(record) + "\n")

            # Create manifest metadata
            manifest = StageManifest(
                stage=stage.value,
                target_percentage=config.target_percentage,
                actual_count=len(records),
                quality_profile=config.quality_profile,
                split_counts={
                    "train": len(splits["train"]),
                    "val": len(splits["val"]),
                    "test": len(splits["test"]),
                },
                manifest_file=config.manifest_filename,
                output_path=str(manifest_path),
            )
            manifests.append(manifest)

            logger.info(
                f"{stage.value}: {len(records)} records "
                f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
            )

        # Write manifests index
        index_path = self.output_dir / "manifests.json"
        with open(index_path, "w") as f:
            json.dump([m.to_dict() for m in manifests], f, indent=2)

        logger.info(f"Wrote {len(manifests)} stage manifests to {self.output_dir}")
        return manifests


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Organize dataset shards into stage-specific manifests")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing JSONL shard files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai/training_data_consolidated/final"),
        help="Output directory for stage manifests (default: ai/training_data_consolidated/final)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    organizer = StageOrganizer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    manifests = organizer.organize()

    print(f"\nOrganized {sum(m.actual_count for m in manifests)} records into {len(manifests)} stages:")
    for m in manifests:
        print(f"  {m.stage}: {m.actual_count} records -> {m.manifest_file}")


if __name__ == "__main__":
    main()
