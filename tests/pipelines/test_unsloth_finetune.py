"""
Tests for Unsloth fine-tuning integration pipeline.

Covers:
- Config loading and validation
- Error handling (ImportError, ValueError, runtime)
- Compliance and safety monitoring hooks
- Logging and extensibility
- No secrets, PEP8, <500 lines, modular

London School TDD: All tests must fail initially for correct reasons.
"""

import os
import tempfile
import json
import logging
import pytest
from unittest.mock import MagicMock, patch

import ai.pipelines.unsloth_finetune as unsloth_mod

class TestUnslothFinetunePipeline:
    # --- Config loading and validation ---

    def test_load_training_config_success(self):
        """Given a valid config file, load_training_config returns the config dict."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            json.dump({"batch_size": 4, "epochs": 2, "learning_rate": 1e-4}, f)
            f.flush()
            path = f.name
        try:
            config = unsloth_mod.load_training_config(config_path=path)
            assert config["batch_size"] == 4
            assert config["epochs"] == 2
            assert config["learning_rate"] == 1e-4
        finally:
            os.remove(path)

    def test_load_training_config_missing_file(self):
        """Given a missing config file, load_training_config raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            unsloth_mod.load_training_config(config_path="nonexistent.json")

    def test_load_training_config_missing_required_fields(self):
        """Given a config missing required fields, load_training_config raises ValueError."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            json.dump({"batch_size": 4}, f)
            f.flush()
            path = f.name
        try:
            with pytest.raises(ValueError):
                unsloth_mod.load_training_config(config_path=path)
        finally:
            os.remove(path)

    def test_load_training_config_overrides(self):
        """Given overrides, load_training_config merges them with file config."""
        with tempfile.NamedTemporaryFile("w+", delete=False) as f:
            json.dump({"batch_size": 4, "epochs": 2, "learning_rate": 1e-4}, f)
            f.flush()
            path = f.name
        try:
            config = unsloth_mod.load_training_config(
                config_path=path, overrides={"epochs": 10}
            )
            assert config["epochs"] == 10
        finally:
            os.remove(path)

    # --- Error handling in finetune_with_unsloth ---

    def test_finetune_with_unsloth_importerror(self):
        """If Unsloth is not installed, finetune_with_unsloth raises ImportError."""
        with patch.object(unsloth_mod, "unsloth", None):
            with pytest.raises(ImportError):
                unsloth_mod.finetune_with_unsloth(tokenized_data=[], config={"batch_size":1,"epochs":1,"learning_rate":1e-4})

    def test_finetune_with_unsloth_valueerror(self):
        """If config is missing required fields, finetune_with_unsloth raises ValueError."""
        with patch.object(unsloth_mod, "unsloth", MagicMock()):
            with pytest.raises(ValueError):
                unsloth_mod.finetune_with_unsloth(tokenized_data=[], config={"batch_size":1})

    def test_finetune_with_unsloth_runtime_error(self):
        """If model loading fails, finetune_with_unsloth logs and raises."""
        mock_unsloth = MagicMock()
        mock_unsloth.load_gguf_model.side_effect = RuntimeError("model load failed")
        with patch.object(unsloth_mod, "unsloth", mock_unsloth):
            with pytest.raises(RuntimeError):
                unsloth_mod.finetune_with_unsloth(tokenized_data=[], config={"batch_size":1,"epochs":1,"learning_rate":1e-4})

    # --- Compliance and safety monitoring integration ---

    def test_finetune_with_unsloth_compliance_and_safety_hooks(self):
        """finetune_with_unsloth calls compliance and safety hooks if provided."""
        mock_unsloth = MagicMock()
        mock_unsloth.load_gguf_model.return_value = MagicMock()
        mock_compliance = MagicMock()
        mock_safety = MagicMock()
        config = {"batch_size":1,"epochs":1,"learning_rate":1e-4}
        with patch.object(unsloth_mod, "unsloth", mock_unsloth):
            unsloth_mod.finetune_with_unsloth(
                tokenized_data=[1,2,3],
                config=config,
                compliance_system=mock_compliance,
                safety_monitor=mock_safety,
            )
        assert mock_compliance.run_pre_training_checks.called
        assert mock_compliance.run_post_training_checks.called
        # SafetyMonitor is not called directly in stub, but should be initialized if not provided

    # --- Logging and extensibility ---

    def test_finetune_with_unsloth_custom_logger(self, caplog):
        """finetune_with_unsloth uses the provided logger for info and error messages."""
        mock_unsloth = MagicMock()
        mock_unsloth.load_gguf_model.return_value = MagicMock()
        config = {"batch_size":1,"epochs":1,"learning_rate":1e-4}
        logger = logging.getLogger("test_logger")
        with patch.object(unsloth_mod, "unsloth", mock_unsloth):
            with caplog.at_level(logging.INFO):
                unsloth_mod.finetune_with_unsloth(tokenized_data=[], config=config, logger=logger)
        assert any("Loaded training config" in m for m in caplog.messages)
        assert any("Fine-tuning completed successfully" in m for m in caplog.messages)

    def test_finetune_with_unsloth_kwargs_extensibility(self):
        """finetune_with_unsloth accepts and passes through extra kwargs."""
        mock_unsloth = MagicMock()
        mock_unsloth.load_gguf_model.return_value = MagicMock()
        config = {"batch_size":1,"epochs":1,"learning_rate":1e-4}
        with patch.object(unsloth_mod, "unsloth", mock_unsloth):
            # Should not raise
            unsloth_mod.finetune_with_unsloth(tokenized_data=[], config=config, extra_param="foo")

# TDD anchor: Add more tests for training loop, metrics, and compliance edge cases as implementation grows.