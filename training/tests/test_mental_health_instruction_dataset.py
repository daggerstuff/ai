"""Tests for mental health instruction dataset builder."""

import json
import tempfile
from pathlib import Path

from ai.training.mental_health_instruction_dataset import (
    MentalHealthInstructionDatasetBuilder,
    MentalHealthTaskType,
    build_default_dataset,
)


def test_build_from_seed_vignettes():
    builder = MentalHealthInstructionDatasetBuilder(seed=42)
    examples = builder.build_from_seed_vignettes(augment_per_vignette=2)
    assert len(examples) > 0
    assert all(ex.task_type for ex in examples)
    assert all(ex.instruction and ex.input and ex.output for ex in examples)


def test_task_type_coverage():
    builder = MentalHealthInstructionDatasetBuilder(seed=42)
    builder.build_from_seed_vignettes(augment_per_vignette=1)
    task_types = {ex.task_type for ex in builder.examples}
    expected = {t.value for t in MentalHealthTaskType}
    assert expected <= task_types


def test_stratified_split():
    builder = MentalHealthInstructionDatasetBuilder(seed=42)
    builder.build_from_seed_vignettes(augment_per_vignette=4)
    train, val = builder.stratified_split(train_ratio=0.8)
    assert len(train) > len(val)
    assert len(train) + len(val) == len(builder.examples)


def test_save_alpaca_format():
    with tempfile.TemporaryDirectory() as tmp:
        builder = MentalHealthInstructionDatasetBuilder(seed=42)
        builder.build_from_seed_vignettes(augment_per_vignette=1)
        train_path, val_path = builder.save(tmp, format="alpaca")
        assert Path(train_path).exists()
        assert Path(val_path).exists()

        with open(train_path) as f:
            first = json.loads(f.readline())
        assert "instruction" in first
        assert "input" in first
        assert "output" in first


def test_build_default_dataset():
    with tempfile.TemporaryDirectory() as tmp:
        train_path, val_path = build_default_dataset(output_dir=tmp, min_examples=500)
        train_count = sum(1 for _ in open(train_path))
        val_count = sum(1 for _ in open(val_path))
        assert train_count + val_count >= 500


def test_to_chat_format():
    builder = MentalHealthInstructionDatasetBuilder(seed=42)
    builder.build_from_seed_vignettes(augment_per_vignette=1)
    chat = builder.examples[0].to_chat(system_prompt="You are a helper.")
    assert "messages" in chat
    assert any(msg["role"] == "system" for msg in chat["messages"])
