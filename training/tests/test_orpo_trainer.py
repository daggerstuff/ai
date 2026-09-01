"""Tests for the ORPO trainer.

The ORPO trainer reuses ``dpo_trainer.load_preference_dataset`` and
``dpo_trainer.CheckpointVerificationCallback``, so those are covered by
``test_dpo_trainer.py``.  Here we test the ORPO-specific pieces:
``save_metrics``, ``build_parser``, ``_build_training_args``, and the
import surface.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from training.orpo_trainer import (
        DEFAULT_BETA,
        DEFAULT_LR,
        DEFAULT_WARMUP_RATIO,
        build_parser,
        save_metrics,
    )
except ImportError:
    pytest.skip("orpo_trainer not importable", allow_module_level=True)

BETA_OVERRIDE = 0.2
BETA_ALTERNATE = 0.15
LR_OVERRIDE = 5e-6
TRAIN_LOSS = 0.42
TRAIN_LOSS_ALT = 0.5
PATIENCE_OVERRIDE = 2
EVAL_STEPS = 100
SAVE_TOTAL_LIMIT = 3
LORA_R = 32
LORA_ALPHA = 64

# ---------------------------------------------------------------------------
# save_metrics
# ---------------------------------------------------------------------------

class TestSaveMetrics:
    """``save_metrics`` writes a JSON report with method, beta, and metrics."""

    def test_writes_valid_json(self, tmp_path: Path):
        output_dir = tmp_path / "orpo_out"
        metrics = {
            "train_loss": TRAIN_LOSS,
            "train_runtime": 3600,
            "beta": DEFAULT_BETA,
            "checkpoint_verification": {"adapter_config.json": True, "adapter_model.safetensors": True},
        }
        save_metrics(output_dir, metrics, beta=DEFAULT_BETA)

        metrics_file = output_dir / "orpo_metrics.json"
        assert metrics_file.exists()

        report = json.loads(metrics_file.read_text(encoding="utf-8"))
        assert report["method"] == "orpo"
        assert report["beta"] == DEFAULT_BETA
        assert report["metrics"]["train_loss"] == TRAIN_LOSS
        assert "generated_at" in report

    def test_creates_output_dir(self, tmp_path: Path):
        output_dir = tmp_path / "nested" / "deep" / "orpo_out"
        save_metrics(output_dir, {"train_loss": 1.0}, beta=BETA_OVERRIDE)
        assert (output_dir / "orpo_metrics.json").exists()

    def test_beta_recorded_separately(self, tmp_path: Path):
        """The ``beta`` argument is recorded at the top level, not just
        inside ``metrics`` — so the report is self-describing even if
        the caller passes a metrics dict without a ``beta`` key."""
        output_dir = tmp_path / "out"
        save_metrics(output_dir, {"train_loss": TRAIN_LOSS_ALT}, beta=BETA_ALTERNATE)
        report = json.loads((output_dir / "orpo_metrics.json").read_text(encoding="utf-8"))
        assert report["beta"] == BETA_ALTERNATE

    def test_extra_fields_merged(self, tmp_path: Path):
        """``extra`` dict fields are merged at the top level of the report."""
        output_dir = tmp_path / "out"
        save_metrics(
            output_dir,
            {"train_loss": TRAIN_LOSS_ALT},
            beta=DEFAULT_BETA,
            extra={"deepspeed_config": "ds_config_zero3.json", "adapter_variant": "dora"},
        )
        report = json.loads((output_dir / "orpo_metrics.json").read_text(encoding="utf-8"))
        assert report["deepspeed_config"] == "ds_config_zero3.json"
        assert report["adapter_variant"] == "dora"

    def test_extra_none_omitted(self, tmp_path: Path):
        """When ``extra`` is None, no extra fields are added."""
        output_dir = tmp_path / "out"
        save_metrics(output_dir, {"train_loss": TRAIN_LOSS_ALT}, beta=DEFAULT_BETA)
        report = json.loads((output_dir / "orpo_metrics.json").read_text(encoding="utf-8"))
        assert "deepspeed_config" not in report
        assert "adapter_variant" not in report

    def test_metrics_file_is_valid_json_with_trailing_newline(self, tmp_path: Path):
        output_dir = tmp_path / "out"
        save_metrics(output_dir, {"train_loss": TRAIN_LOSS_ALT}, beta=DEFAULT_BETA)
        raw = (output_dir / "orpo_metrics.json").read_text(encoding="utf-8")
        assert raw.endswith("\n")
        json.loads(raw)  # Should not raise

# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    """``build_parser`` produces an argparse parser with ORPO-specific args."""

    def test_required_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "data.jsonl",
            "--base_model_checkpoint", "Qwen/Qwen2.5-32B",
            "--output_dir", "out/",
        ])
        assert args.data_path == "data.jsonl"
        assert args.base_model_checkpoint == "Qwen/Qwen2.5-32B"
        assert args.output_dir == "out/"

    def test_beta_default(self):
        """Appendix C specifies beta=0.1 as the default."""
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.beta == DEFAULT_BETA

    def test_beta_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--beta", str(BETA_OVERRIDE),
        ])
        assert args.beta == BETA_OVERRIDE

    def test_learning_rate_default(self):
        """ORPO uses a lower learning rate than SFT (5e-6 vs 2e-5)."""
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.learning_rate == DEFAULT_LR == LR_OVERRIDE

    def test_warmup_ratio_default(self):
        """Appendix C: warmup_ratio 0.1."""
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.warmup_ratio == DEFAULT_WARMUP_RATIO

    def test_lr_scheduler_default(self):
        """Appendix C: cosine schedule."""
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.lr_scheduler_type == "cosine"

    @pytest.mark.parametrize("sched", ["cosine", "linear", "constant", "constant_with_warmup"])
    def test_lr_scheduler_choices(self, sched):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--lr_scheduler_type", sched,
        ])
        assert args.lr_scheduler_type == sched

    def test_deepspeed_arg(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--deepspeed", "ai/training/configs/ds_config_zero3.json",
        ])
        assert args.deepspeed == "ai/training/configs/ds_config_zero3.json"

    def test_deepspeed_default_none(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.deepspeed is None

    def test_gradient_checkpointing_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--gradient_checkpointing",
        ])
        assert args.gradient_checkpointing is True

    def test_gradient_checkpointing_default_false(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.gradient_checkpointing is False

    def test_flash_attention_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--flash_attention",
        ])
        assert args.flash_attention is True

    def test_early_stopping_patience_default(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.early_stopping_patience == 0

    def test_early_stopping_patience_override(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--early_stopping_patience", str(PATIENCE_OVERRIDE),
        ])
        assert args.early_stopping_patience == PATIENCE_OVERRIDE

    def test_eval_strategy_default_no(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.eval_strategy == "no"

    def test_eval_strategy_steps(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--eval_strategy", "steps", "--eval_steps", str(EVAL_STEPS),
        ])
        assert args.eval_strategy == "steps"
        assert args.eval_steps == EVAL_STEPS

    def test_save_total_limit_default(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
        ])
        assert args.save_total_limit == SAVE_TOTAL_LIMIT

    def test_wandb_args(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--wandb_project", "pixelated-empathy-orpo",
            "--wandb_entity", "team",
            "--wandb_run_name", "orpo-run-1",
        ])
        assert args.wandb_project == "pixelated-empathy-orpo"
        assert args.wandb_entity == "team"
        assert args.wandb_run_name == "orpo-run-1"

    def test_use_dora_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--use_dora",
        ])
        assert args.use_dora is True

    def test_use_vera_flag(self):
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--use_vera",
        ])
        assert args.use_vera is True

    def test_lora_args_inherited(self):
        """LoRA args from ``shared_config.add_lora_args`` are present."""
        parser = build_parser()
        args = parser.parse_args([
            "--data_path", "d", "--base_model_checkpoint", "m", "--output_dir", "o",
            "--lora_r", str(LORA_R), "--lora_alpha", str(LORA_ALPHA),
        ])
        assert args.lora_r == LORA_R
        assert args.lora_alpha == LORA_ALPHA

    def test_missing_required_arg_exits(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--data_path", "d"])

# ---------------------------------------------------------------------------
# Import surface — verify ORPO trainer reuses DPO components
# ---------------------------------------------------------------------------

def _import_from_dpo_deps(module: str, name: str):
    """Import ``name`` preferring ``training.*``, falling back to
    ``ai.training.*`` when the package root differs."""
    try:
        module_obj = __import__(f"training.{module}", fromlist=[name])
    except ImportError:
        module_obj = __import__(f"ai.training.{module}", fromlist=[name])
    return getattr(module_obj, name)

class TestImportSurface:
    """Verify the ORPO trainer reuses DPO components correctly."""

    def test_load_preference_dataset_reused(self):
        """``load_preference_dataset`` should be importable from the ORPO
        module's dependencies (i.e. from ``dpo_trainer``)."""
        assert callable(_import_from_dpo_deps("dpo_trainer", "load_preference_dataset"))

    def test_checkpoint_callback_reused(self):
        callback_cls = _import_from_dpo_deps("dpo_trainer", "CheckpointVerificationCallback")
        callback = callback_cls()
        assert hasattr(callback, "verify")

    def test_shared_config_reused(self):
        assert callable(_import_from_dpo_deps("shared_config", "shared_qlora_config"))

    def test_log_token_length_distribution_reused(self):
        assert callable(_import_from_dpo_deps("shared_config", "log_token_length_distribution"))

    def test_run_orpo_callable(self):
        assert callable(_import_from_dpo_deps("orpo_trainer", "run_orpo"))

    def test_build_training_args_callable(self):
        assert callable(_import_from_dpo_deps("orpo_trainer", "_build_training_args"))

    def test_setup_wandb_callable(self):
        assert callable(_import_from_dpo_deps("orpo_trainer", "_setup_wandb"))
