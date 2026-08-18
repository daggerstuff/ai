"""Tests for stage_organizer module."""

from __future__ import annotations

import json

import pytest

from ai.pipelines.orchestration.stage_organizer import (
    DEFAULT_STAGE_CONFIGS,
    Stage,
    StageConfig,
    StageManifest,
    StageOrganizer,
    classify_record,
    enforce_quotas,
    split_dataset,
)


class TestClassification:
    """Test record classification into training stages."""

    def test_classify_stage5_safety(self):
        """Records with safety sources go to Stage 5."""
        record = {"source": "crisis_intervention", "metadata": {"topic_tags": ["crisis"]}}
        assert classify_record(record) == Stage.STAGE5_SAFETY

    def test_classify_stage5_safety_tags(self):
        """Records with safety tags go to Stage 5."""
        record = {"source": "general", "metadata": {"topic_tags": ["self_harm", "suicide"]}}
        assert classify_record(record) == Stage.STAGE5_SAFETY

    def test_classify_stage4_voice_persona(self):
        """Records with voice/persona sources go to Stage 4."""
        record = {"source": "pixel_voice", "metadata": {"topic_tags": []}}
        assert classify_record(record) == Stage.STAGE4_VOICE_PERSONA

    def test_classify_stage4_voice_tags(self):
        """Records with voice/persona tags go to Stage 4."""
        record = {"source": "general", "metadata": {"topic_tags": ["persona", "dual_persona"]}}
        assert classify_record(record) == Stage.STAGE4_VOICE_PERSONA

    def test_classify_stage3_edge(self):
        """Records with edge case sources go to Stage 3."""
        record = {"source": "adversarial", "metadata": {"topic_tags": []}}
        assert classify_record(record) == Stage.STAGE3_EDGE_STRESS_TEST

    def test_classify_stage3_edge_tags(self):
        """Records with edge case tags go to Stage 3."""
        record = {"source": "general", "metadata": {"topic_tags": ["jailbreak", "red_team"]}}
        assert classify_record(record) == Stage.STAGE3_EDGE_STRESS_TEST

    def test_classify_stage2_therapeutic(self):
        """Records with therapeutic modality go to Stage 2."""
        record = {"source": "clinical", "metadata": {"therapeutic_modality": "cbt", "topic_tags": []}}
        assert classify_record(record) == Stage.STAGE2_THERAPEUTIC_EXPERTISE

    def test_classify_stage2_therapeutic_tags(self):
        """Records with therapeutic tags go to Stage 2."""
        record = {"source": "general", "metadata": {"topic_tags": ["therapy", "psychotherapy"]}}
        assert classify_record(record) == Stage.STAGE2_THERAPEUTIC_EXPERTISE

    def test_classify_stage1_foundation(self):
        """Records with foundation tags go to Stage 1."""
        record = {"source": "general", "metadata": {"topic_tags": ["psychology", "mental_health"]}}
        assert classify_record(record) == Stage.STAGE1_FOUNDATION

    def test_classify_stage1_default(self):
        """Unclassified records default to Stage 1."""
        record = {"source": "unknown", "metadata": {"topic_tags": []}}
        assert classify_record(record) == Stage.STAGE1_FOUNDATION

    def test_classify_priority_safety_over_voice(self):
        """Safety takes priority over voice/persona."""
        record = {"source": "crisis_intervention", "metadata": {"topic_tags": ["persona"]}}
        assert classify_record(record) == Stage.STAGE5_SAFETY


class TestSplitDataset:
    """Test 80/10/10 dataset splitting."""

    def test_split_ratios(self):
        """Split produces correct 80/10/10 ratios."""
        records = [{"id": i} for i in range(100)]
        splits = split_dataset(records, train_ratio=0.80, val_ratio=0.10)

        assert len(splits["train"]) == 80
        assert len(splits["val"]) == 10
        assert len(splits["test"]) == 10

    def test_split_no_overlap(self):
        """Split produces non-overlapping sets."""
        records = [{"id": i} for i in range(100)]
        splits = split_dataset(records)

        train_ids = {r["id"] for r in splits["train"]}
        val_ids = {r["id"] for r in splits["val"]}
        test_ids = {r["id"] for r in splits["test"]}

        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_split_reproducible(self):
        """Split is reproducible with same seed."""
        records = [{"id": i} for i in range(100)]
        splits1 = split_dataset(records, seed=42)
        splits2 = split_dataset(records, seed=42)

        assert splits1["train"] == splits2["train"]
        assert splits1["val"] == splits2["val"]
        assert splits1["test"] == splits2["test"]

    def test_split_small_dataset(self):
        """Split handles small datasets gracefully."""
        records = [{"id": i} for i in range(5)]
        splits = split_dataset(records)

        # With 5 records: 4 train, 0 val, 1 test (due to int rounding)
        assert len(splits["train"]) == 4
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 5


class TestQuotaEnforcement:
    """Test stage quota enforcement."""

    def test_enforce_quotas_redistributes_overflow_no_drop(self):
        """Quotas redistribute overflow; over-represented stages are never capped (P0-4)."""
        stage_records = {
            Stage.STAGE1_FOUNDATION: [{"id": i} for i in range(100)],
            Stage.STAGE2_THERAPEUTIC_EXPERTISE: [{"id": i} for i in range(10)],
        }
        configs = {
            Stage.STAGE1_FOUNDATION: StageConfig(Stage.STAGE1_FOUNDATION, 0.50),
            Stage.STAGE2_THERAPEUTIC_EXPERTISE: StageConfig(Stage.STAGE2_THERAPEUTIC_EXPERTISE, 0.50),
        }

        result = enforce_quotas(stage_records, total_records=110, configs=configs)

        # Stage 1 exceeds its 55-record target, but overflow is retained, not dropped.
        assert len(result[Stage.STAGE1_FOUNDATION]) == 100
        # Stage 2 stays at 10 (under quota).
        assert len(result[Stage.STAGE2_THERAPEUTIC_EXPERTISE]) == 10
        # No record is silently discarded.
        total = sum(len(r) for r in result.values())
        assert total == 110

        # Per-stage ID integrity: every input record must appear in its
        # original stage (no silent reshuffling or replacement).
        stage1_ids = {r["id"] for r in result[Stage.STAGE1_FOUNDATION]}
        stage2_ids = {r["id"] for r in result[Stage.STAGE2_THERAPEUTIC_EXPERTISE]}
        assert stage1_ids == {i for i in range(100)}
        assert stage2_ids == {i for i in range(10)}

    def test_enforce_quotas_preserves_underrepresented(self):
        """Quotas preserve under-represented stages."""
        stage_records = {
            Stage.STAGE1_FOUNDATION: [{"id": i} for i in range(20)],
        }
        configs = {
            Stage.STAGE1_FOUNDATION: StageConfig(Stage.STAGE1_FOUNDATION, 0.50),
        }

        result = enforce_quotas(stage_records, total_records=100, configs=configs)

        # Stage 1 has 20 records, quota is 50, so all 20 should remain
        assert len(result[Stage.STAGE1_FOUNDATION]) == 20


class TestStageOrganizer:
    """Test full StageOrganizer pipeline."""

    @pytest.fixture
    def sample_shards(self, tmp_path):
        """Create sample JSONL shards for testing."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        records = [
            {"source": "general", "metadata": {"topic_tags": ["psychology"]}, "text": "sample 1"},
            {"source": "general", "metadata": {"topic_tags": ["therapy"]}, "text": "sample 2"},
            {"source": "adversarial", "metadata": {"topic_tags": []}, "text": "sample 3"},
            {"source": "pixel_voice", "metadata": {"topic_tags": []}, "text": "sample 4"},
            {"source": "crisis_intervention", "metadata": {"topic_tags": []}, "text": "sample 5"},
        ]

        shard_path = input_dir / "shard_001.jsonl"
        with open(shard_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

        return input_dir

    def test_organizer_loads_shards(self, sample_shards, tmp_path):
        """Organizer loads and classifies records from shards."""
        output_dir = tmp_path / "output"
        organizer = StageOrganizer(input_dir=sample_shards, output_dir=output_dir)

        total = organizer.load_shards()

        assert total == 5
        assert len(organizer._stage_records[Stage.STAGE1_FOUNDATION]) == 1
        assert len(organizer._stage_records[Stage.STAGE2_THERAPEUTIC_EXPERTISE]) == 1
        assert len(organizer._stage_records[Stage.STAGE3_EDGE_STRESS_TEST]) == 1
        assert len(organizer._stage_records[Stage.STAGE4_VOICE_PERSONA]) == 1
        assert len(organizer._stage_records[Stage.STAGE5_SAFETY]) == 1

    def test_organizer_writes_manifests(self, sample_shards, tmp_path):
        """Organizer writes manifest files to output directory."""
        output_dir = tmp_path / "output"
        organizer = StageOrganizer(input_dir=sample_shards, output_dir=output_dir)

        manifests = organizer.organize()

        # Check at least some manifest files exist (small sample may not have all stages)
        assert (output_dir / "manifests.json").exists()
        assert len(manifests) > 0

        # Check that manifests list matches created files
        for manifest in manifests:
            assert (output_dir / manifest.manifest_file).exists()

    def test_organizer_manifest_structure(self, sample_shards, tmp_path):
        """Organizer produces correct manifest structure."""
        output_dir = tmp_path / "output"
        organizer = StageOrganizer(input_dir=sample_shards, output_dir=output_dir)

        manifests = organizer.organize()

        for manifest in manifests:
            assert isinstance(manifest.stage, str)
            assert isinstance(manifest.target_percentage, float)
            assert isinstance(manifest.actual_count, int)
            assert isinstance(manifest.quality_profile, dict)
            assert isinstance(manifest.split_counts, dict)
            assert "train" in manifest.split_counts
            assert "val" in manifest.split_counts
            assert "test" in manifest.split_counts

    def test_organizer_split_files(self, sample_shards, tmp_path):
        """Organizer writes train/val/test split files."""
        output_dir = tmp_path / "output"
        organizer = StageOrganizer(input_dir=sample_shards, output_dir=output_dir)

        manifests = organizer.organize()

        # Check split files exist for stages that have data
        for manifest in manifests:
            if manifest.actual_count > 0:
                stage_base = manifest.manifest_file.replace(".jsonl", "")
                train_file = output_dir / f"{stage_base}_train.jsonl"
                val_file = output_dir / f"{stage_base}_val.jsonl"
                test_file = output_dir / f"{stage_base}_test.jsonl"

                # At least one split file should exist
                assert train_file.exists() or val_file.exists() or test_file.exists()

    def test_organizer_handles_empty_input(self, tmp_path):
        """Organizer raises error for empty input directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        organizer = StageOrganizer(input_dir=input_dir, output_dir=output_dir)

        with pytest.raises(FileNotFoundError):
            organizer.load_shards()

    def test_organizer_handles_missing_input(self, tmp_path):
        """Organizer raises error for missing input directory."""
        input_dir = tmp_path / "nonexistent"
        output_dir = tmp_path / "output"

        organizer = StageOrganizer(input_dir=input_dir, output_dir=output_dir)

        with pytest.raises(FileNotFoundError):
            organizer.load_shards()


class TestStageConfig:
    """Test StageConfig dataclass."""

    def test_manifest_filename(self):
        """StageConfig generates correct manifest filename."""
        config = StageConfig(Stage.STAGE1_FOUNDATION, 0.40)
        assert config.manifest_filename == "MASTER_STAGE_1.jsonl"

        config = StageConfig(Stage.STAGE5_SAFETY, 0.05)
        assert config.manifest_filename == "MASTER_STAGE_5.jsonl"

    def test_default_configs(self):
        """Default configs have correct target percentages."""
        assert DEFAULT_STAGE_CONFIGS[Stage.STAGE1_FOUNDATION].target_percentage == 0.35
        assert DEFAULT_STAGE_CONFIGS[Stage.STAGE2_THERAPEUTIC_EXPERTISE].target_percentage == 0.25
        assert DEFAULT_STAGE_CONFIGS[Stage.STAGE3_EDGE_STRESS_TEST].target_percentage == 0.20
        assert DEFAULT_STAGE_CONFIGS[Stage.STAGE4_VOICE_PERSONA].target_percentage == 0.15
        assert DEFAULT_STAGE_CONFIGS[Stage.STAGE5_SAFETY].target_percentage == 0.05

        # Total should be 100%
        total = sum(c.target_percentage for c in DEFAULT_STAGE_CONFIGS.values())
        assert total == 1.0


class TestStageManifest:
    """Test StageManifest dataclass."""

    def test_to_dict(self):
        """StageManifest converts to dict correctly."""
        manifest = StageManifest(
            stage="stage1_foundation",
            target_percentage=0.40,
            actual_count=100,
            quality_profile={"empathy_floor": 0.70},
            split_counts={"train": 80, "val": 10, "test": 10},
            manifest_file="MASTER_STAGE_1.jsonl",
            output_path="/output/MASTER_STAGE_1.jsonl",
        )

        d = manifest.to_dict()

        assert d["stage"] == "stage1_foundation"
        assert d["target_percentage"] == 0.40
        assert d["actual_count"] == 100
        assert d["split_counts"]["train"] == 80
