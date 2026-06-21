#!/usr/bin/env python3
"""Unit tests for SFT script improvements from Phase 1."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from training.pixelated_production_pilot import (
    CheckpointVerificationCallback,
    HubConfig,
    RunConfig,
    _build_arg_parser,
    _maybe_push_to_hub,
    _validate_numeric_args,
    check_disk_space,
    safe_path,
)
from training.shared_config import build_lora_config

# ---------------------------------------------------------------------------
# _build_arg_parser / --max_seq_length validation
# ---------------------------------------------------------------------------


class TestSftSeqLengthArgparse:
    def test_max_seq_length_511_raises_system_exit(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--data_path", "/tmp/data.jsonl",
                "--output_dir", "/tmp/output",
                "--max_seq_length", "511",
            ])

    def test_max_seq_length_8193_raises_system_exit(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--data_path", "/tmp/data.jsonl",
                "--output_dir", "/tmp/output",
                "--max_seq_length", "8193",
            ])

    def test_max_seq_length_2048_accepted(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--data_path", "/tmp/data.jsonl",
            "--output_dir", "/tmp/output",
            "--max_seq_length", "2048",
        ])
        assert args.max_seq_length == 2048


# ---------------------------------------------------------------------------
# build_lora_config (via _build_arg_parser args)
# ---------------------------------------------------------------------------


class TestLoraTargetModules:
    def test_gate_proj_up_proj_down_proj_accepted(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        args = argparse.Namespace(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_bias="none",
            lora_target_modules="gate_proj,up_proj,down_proj",
        )
        lora_config = build_lora_config(args)
        assert {module.strip() for module in (lora_config.target_modules or [])} == {
            "gate_proj",
            "up_proj",
            "down_proj",
        }


# ---------------------------------------------------------------------------
# safe_path
# ---------------------------------------------------------------------------


class TestSafePath:
    def test_absolute_path_within_workspace(self) -> None:
        """Absolute path within WORKSPACE_ROOT is accepted."""
        workspace_root = Path("/tmp/test_workspace")
        workspace_root.mkdir(parents=True, exist_ok=True)
        inner = workspace_root / "subdir"
        inner.mkdir(parents=True, exist_ok=True)
        try:
            with patch("training.pixelated_production_pilot.WORKSPACE_ROOT", workspace_root):
                result = safe_path(str(inner))
            assert result == inner.resolve()
        finally:
            import shutil
            shutil.rmtree(workspace_root, ignore_errors=True)

    def test_relative_path_resolved_against_workspace(self) -> None:
        """Relative paths are resolved against WORKSPACE_ROOT."""
        workspace_root = Path("/tmp/test_workspace2")
        workspace_root.mkdir(parents=True, exist_ok=True)
        try:
            with patch("training.pixelated_production_pilot.WORKSPACE_ROOT", workspace_root):
                result = safe_path("data/file.json")
            assert str(result).startswith(str(workspace_root.resolve()))
        finally:
            import shutil
            shutil.rmtree(workspace_root, ignore_errors=True)

    def test_path_outside_workspace_raises_permission_error(self) -> None:
        """Paths outside WORKSPACE_ROOT raise PermissionError."""
        workspace_root = Path("/tmp/ws_restricted")
        workspace_root.mkdir(parents=True, exist_ok=True)
        try:
            with patch("training.pixelated_production_pilot.WORKSPACE_ROOT", workspace_root):
                with pytest.raises(PermissionError, match="outside workspace"):
                    safe_path("/etc/passwd")
        finally:
            import shutil
            shutil.rmtree(workspace_root, ignore_errors=True)

    def test_resolved_path_escapes_symlink_raises(self, tmp_path) -> None:
        """A resolved symlink that escapes WORKSPACE_ROOT raises PermissionError."""
        outer = tmp_path / "outer"
        escape_target = tmp_path / "escape_target"
        escape_target.mkdir()
        outer.mkdir()
        link = outer / "link"
        link.symlink_to(escape_target, target_is_directory=True)
        workspace_root = outer.resolve()
        with patch("training.pixelated_production_pilot.WORKSPACE_ROOT", workspace_root):
            with pytest.raises(PermissionError, match="outside workspace"):
                safe_path(str(link))


# ---------------------------------------------------------------------------
# check_disk_space
# ---------------------------------------------------------------------------


class TestCheckDiskSpace:
    def test_sufficient_disk_logs_debug(self, caplog) -> None:
        """When enough space is available, a debug message is logged."""
        caplog.set_level(logging.DEBUG)
        check_disk_space(Path("/tmp"), required_gb=0.001)
        assert any("Disk space OK" in msg for msg in caplog.messages)

    def test_low_disk_logs_warning(self, caplog) -> None:
        """When disk space is below threshold, a warning is logged."""
        caplog.set_level(logging.WARNING)
        check_disk_space(Path("/tmp"), required_gb=1_000_000.0)
        assert any("Low disk space" in msg for msg in caplog.messages)

    def test_invalid_path_logs_warning(self, caplog) -> None:
        """An invalid path logs a warning, does not crash."""
        caplog.set_level(logging.WARNING)
        check_disk_space(Path("/nonexistent_path_xyz123"), required_gb=5.0)
        assert any("Could not check disk space" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# _validate_numeric_args (pure logic, no ML deps)
# ---------------------------------------------------------------------------


class TestValidateNumericArgs:
    def test_valid_args_passes(self) -> None:
        args = argparse.Namespace(
            max_steps=100,
            learning_rate=1e-4,
            batch_size=2,
            lora_r=8,
            lora_dropout=0.05,
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            weight_decay=0.01,
        )
        _validate_numeric_args(args)  # should not raise

    def test_max_steps_zero_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=0, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_max_steps_negative_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=-1, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_learning_rate_zero_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=0, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_learning_rate_negative_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=-1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_batch_size_zero_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=0, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_lora_r_zero_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=0,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_lora_dropout_negative_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=-0.1, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_lora_dropout_equal_one_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=1.0, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_warmup_ratio_negative_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=-0.1, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_warmup_ratio_equal_one_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=1.0, max_grad_norm=1.0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_max_grad_norm_zero_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=0, weight_decay=0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)

    def test_weight_decay_negative_raises(self) -> None:
        args = argparse.Namespace(
            max_steps=100, learning_rate=1e-4, batch_size=2, lora_r=8,
            lora_dropout=0.05, warmup_ratio=0.1, max_grad_norm=1.0, weight_decay=-0.01,
        )
        with pytest.raises(SystemExit):
            _validate_numeric_args(args)


# ---------------------------------------------------------------------------
# _tokenizer_supports_gen_prompt
# ---------------------------------------------------------------------------


class TestTokenizerSupportsGenPrompt:
    def test_supports_add_generation_prompt(self) -> None:
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock()
        # Simulate a signature that includes add_generation_prompt
        import inspect
        def fake_sig(**_k):
            return inspect.Parameter("add_generation_prompt", inspect.Parameter.KEYWORD_ONLY)

        tokenizer.apply_chat_template.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY),
             inspect.Parameter("add_generation_prompt", inspect.Parameter.KEYWORD_ONLY, default=False)]
        )
        from training.pixelated_production_pilot import _tokenizer_supports_gen_prompt
        assert _tokenizer_supports_gen_prompt(tokenizer) is True

    def test_does_not_support_add_generation_prompt(self) -> None:
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock()
        import inspect
        tokenizer.apply_chat_template.__signature__ = inspect.Signature(
            [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)]
        )
        from training.pixelated_production_pilot import _tokenizer_supports_gen_prompt
        assert _tokenizer_supports_gen_prompt(tokenizer) is False

    def test_fallback_on_exception(self) -> None:
        """When inspect raises, _tokenizer_supports_gen_prompt returns False."""
        from unittest.mock import MagicMock
        tokenizer = MagicMock()
        tokenizer.apply_chat_template = MagicMock(side_effect=RuntimeError("boom"))
        from training.pixelated_production_pilot import _tokenizer_supports_gen_prompt
        assert _tokenizer_supports_gen_prompt(tokenizer) is False


# ---------------------------------------------------------------------------
# SecureLogHandler
# ---------------------------------------------------------------------------


class TestSecureLogHandler:
    def test_sensitive_pattern_redacted(self) -> None:
        """Sensitive keys are redacted in log messages."""
        from training.pixelated_production_pilot import SecureLogHandler
        handler = SecureLogHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="api_key=sk-1234567890abcdef",
            args=(), exc_info=None,
        )
        handler.emit(record)
        assert "sk-1234567890abcdef" not in record.msg
        assert "api_key=[REDACTED]" in record.msg

    def test_sensitive_pattern_args_redacted(self) -> None:
        """Sensitive keys in tuple args are redacted."""
        from training.pixelated_production_pilot import SecureLogHandler
        handler = SecureLogHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Secret: %s",
            args=("token=ghp_abcdef12345",),
            exc_info=None,
        )
        handler.emit(record)
        assert "ghp_abcdef12345" not in str(record.args)
        assert any("[REDACTED]" in str(a) for a in record.args)

    def test_emit_exception_handled_gracefully(self) -> None:
        """If emit raises, handleError is called (does not crash)."""
        from training.pixelated_production_pilot import SecureLogHandler
        handler = SecureLogHandler()
        # Force an error by passing a non-serializable record
        class BadRecord:
            msg = 42  # not a string
            args = ()
        record = BadRecord()
        # Should not raise — handleError catches it
        handler.emit(record)  # no exception expected

    def test_missing_attribute_args_tuple_redacted(self) -> None:
        """Args that are strings get redacted."""
        from training.pixelated_production_pilot import SecureLogHandler
        handler = SecureLogHandler()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Connection using %s",
            args=("password=supersecret123",),
            exc_info=None,
        )
        handler.emit(record)
        assert any("password=[REDACTED]" in str(a) for a in record.args)


# ---------------------------------------------------------------------------
# safe_path — edge cases
# ---------------------------------------------------------------------------


class TestSafePathEdgeCases:
    def test_path_with_null_character_raises_permission_error(self) -> None:
        """A path with an embedded null raises PermissionError via resolution failure."""
        with pytest.raises(PermissionError, match="Path resolution error"):
            safe_path("\x00")


# ---------------------------------------------------------------------------
# HubConfig
# ---------------------------------------------------------------------------


class TestHubConfig:
    def test_default_config(self) -> None:
        """Default HubConfig has push_to_hub=False and hub_model_id=None."""
        cfg = HubConfig()
        assert cfg.push_to_hub is False
        assert cfg.hub_model_id is None

    def test_custom_config(self) -> None:
        """Custom HubConfig preserves the given values."""
        cfg = HubConfig(push_to_hub=True, hub_model_id="org/my-model")
        assert cfg.push_to_hub is True
        assert cfg.hub_model_id == "org/my-model"


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


class TestRunConfig:
    def test_default_config(self) -> None:
        """Default RunConfig has hub_config=None and skip_final_eval=False."""
        cfg = RunConfig()
        assert cfg.hub_config is None
        assert cfg.skip_final_eval is False

    def test_custom_config(self) -> None:
        """Custom RunConfig preserves given values."""
        hub = HubConfig(push_to_hub=True, hub_model_id="org/my-model")
        cfg = RunConfig(hub_config=hub, skip_final_eval=True)
        assert cfg.hub_config is not None
        assert cfg.hub_config.push_to_hub is True
        assert cfg.skip_final_eval is True


# ---------------------------------------------------------------------------
# _maybe_push_to_hub (pure-logic early-return paths, no ML deps)
# ---------------------------------------------------------------------------


class TestMaybePushToHub:
    def test_none_hub_config_returns_early(self) -> None:
        """When hub_config is None, _maybe_push_to_hub returns immediately."""
        from unittest.mock import MagicMock

        trainer = MagicMock()
        tokenizer = MagicMock()
        # Should not raise — early return before any Hub interaction
        _maybe_push_to_hub(trainer, tokenizer, None)

    def test_push_to_hub_false_returns_early(self) -> None:
        """When push_to_hub is False, returns immediately."""
        from unittest.mock import MagicMock

        trainer = MagicMock()
        tokenizer = MagicMock()
        hub_cfg = HubConfig(push_to_hub=False, hub_model_id="org/model")
        _maybe_push_to_hub(trainer, tokenizer, hub_cfg)

    def test_missing_hub_model_id_returns_early(self) -> None:
        """When hub_model_id is None, returns immediately."""
        from unittest.mock import MagicMock

        trainer = MagicMock()
        tokenizer = MagicMock()
        hub_cfg = HubConfig(push_to_hub=True, hub_model_id=None)
        _maybe_push_to_hub(trainer, tokenizer, hub_cfg)


# ---------------------------------------------------------------------------
# CheckpointVerificationCallback
# ---------------------------------------------------------------------------


class TestCheckpointVerificationCallback:
    def test_on_save_no_checkpoint_dir(self, caplog) -> None:
        """When checkpoint_dir doesn't exist, a warning is logged."""
        from transformers import TrainerControl, TrainerState, TrainingArguments

        caplog.set_level(logging.WARNING)
        callback = CheckpointVerificationCallback()
        args = TrainingArguments(output_dir="/tmp/nonexistent_checkpoint_test_xyz", report_to="none")
        state = TrainerState(global_step=42)
        control = TrainerControl()

        callback.on_save(args, state, control)

        assert any("Expected checkpoint dir not found" in msg for msg in caplog.messages)

    def test_on_save_missing_required_files(self, tmp_path, caplog) -> None:
        """When required files are missing, an error is logged and temp dir removed."""
        from transformers import TrainerControl, TrainerState, TrainingArguments

        caplog.set_level(logging.ERROR)
        callback = CheckpointVerificationCallback()
        checkpoint_dir = tmp_path / "checkpoint-1"
        checkpoint_dir.mkdir(parents=True)

        args = TrainingArguments(output_dir=str(tmp_path), report_to="none")
        state = TrainerState(global_step=1)
        control = TrainerControl()

        callback.on_save(args, state, control)
        assert any("incomplete" in msg for msg in caplog.messages)
