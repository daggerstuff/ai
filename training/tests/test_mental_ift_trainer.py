"""Tests for mental health IFT trainer."""

import tempfile
from pathlib import Path

from ai.training.mental_ift_trainer import IFTConfig, MentalHealthIFTTrainer


def test_config_defaults():
    config = IFTConfig()
    assert config.base_model
    assert config.output_dir
    assert config.lora_r > 0


def test_trainer_initialization():
    config = IFTConfig(output_dir="./test_ift_output")
    trainer = MentalHealthIFTTrainer(config)
    assert trainer.config == config
    assert trainer.model is None


def test_dataset_building():
    config = IFTConfig()
    trainer = MentalHealthIFTTrainer(config)
    dataset = trainer.load_or_build_dataset()
    assert len(dataset) > 0
    assert "instruction" in dataset.column_names
    assert "input" in dataset.column_names
    assert "output" in dataset.column_names


def test_curriculum_sorting():
    config = IFTConfig(curriculum_learning=True)
    trainer = MentalHealthIFTTrainer(config)
    trainer.load_or_build_dataset()
    sorted_dataset = trainer._apply_curriculum(trainer.load_or_build_dataset())
    assert len(sorted_dataset) > 0
