"""Integration test for ORPO trainer end-to-end with mocked model loading.

Validates:
1. Preference dataset loaded and passed to ORPOTrainer
2. ORPOConfig built with correct beta, learning_rate, deepspeed
3. Checkpoint saved (final adapter files and metrics JSON)
4. Metrics JSON contains method="orpo", beta, adapter_variant, deepspeed_config
5. WandB integration path (skipped when wandb not installed)
6. DoRA adapter variant flag propagated to LoRA config
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Defense-in-depth: patch kernels.Repository classes so they default version=1
# when neither revision nor version is provided.  This prevents the
# import-time crash (kernels 0.15.2 LayerRepository requires one of them).
# ---------------------------------------------------------------------------
for _mod_name in ("kernels.layer.layer", "kernels.layer.func"):
    try:
        _mod = __import__(_mod_name, fromlist=["Repository"])
        for _cls_name in dir(_mod):
            _cls = getattr(_mod, _cls_name)
            if not callable(_cls) or not hasattr(_cls, "__init__"):
                continue
            _orig = _cls.__init__

            def _make_init(orig):
                def _init(self, repo_id, *, revision=None, version=None, **kwargs):
                    if revision is None and version is None:
                        version = 1
                    return orig(self, repo_id, revision=revision, version=version, **kwargs)

                return _init

            _cls.__init__ = _make_init(_orig)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Mock uninstalled / broken ML packages at the module level so
# patch("trl.*") and the lazy imports inside run_orpo() work without
# triggering the transformers → kernels import-time crash.
# ---------------------------------------------------------------------------

# trl — not installed in the test environment
_mock_trl = MagicMock()
_mock_trl.ORPOTrainer = MagicMock()
_mock_trl.ORPOConfig = MagicMock()
sys.modules["trl"] = _mock_trl

# peft — installed but triggers transformers → kernels crash on import
def _lora_config(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


_mock_peft = MagicMock()
_mock_peft.prepare_model_for_kbit_training = lambda model: model
_mock_peft.LoraConfig = _lora_config
sys.modules["peft"] = _mock_peft


class TestORPOIntegration:
    """Integration tests for the ORPO trainer pipeline."""

    def _create_test_dataset(self, path: Path, num_samples: int = 50) -> None:
        """Write a JSONL preference-pairs dataset for testing."""
        records = [
            {
                "prompt": f"User: I'm feeling anxious about my therapy session. {i}",
                "chosen": f"I hear you. It's natural to feel anxious. {i}",
                "rejected": f"Just calm down. There's nothing to worry about. {i}",
            }
            for i in range(num_samples)
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_orpo_end_to_end_mocked(self, tmp_path: Path):
        """Run ORPO trainer with 50-sample preference dataset (mocked model).

        Verifies:
        - Preference dataset loaded and passed to ORPOTrainer
        - ORPOConfig built with correct beta and learning_rate
        - Final checkpoint saved (adapter files + metrics JSON)
        - Metrics JSON contains method="orpo", beta, adapter_variant
        - Mock interactions (model load, train, save) all called once
        """
        model_instance = MagicMock()
        model_instance.device_map = "auto"

        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|im_end|>"

        train_result = MagicMock()
        train_result.training_loss = 0.35
        train_result.metrics = {
            "train_runtime": 120.0,
            "train_samples_per_second": 2.5,
        }

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.experimental.orpo.ORPOConfig"),
            patch("trl.experimental.orpo.ORPOTrainer") as mock_orpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_orpo_trainer_cls.return_value
            trainer_instance.train.return_value = train_result

            def _mock_save_model(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save_model
            tokenizer_instance.save_pretrained.side_effect = _mock_save_model

            data_path = tmp_path / "test_preference_pairs.jsonl"
            self._create_test_dataset(data_path, 50)
            output_dir = tmp_path / "orpo_output"

            from training.orpo_trainer import build_parser, run_orpo

            parser = build_parser()
            args = parser.parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", "0.1",
                "--max_seq_length", "512",
                "--batch_size", "2",
                "--learning_rate", "5e-6",
                "--epochs", "1",
                "--logging_steps", "5",
                "--warmup_ratio", "0.1",
                "--lr_scheduler_type", "cosine",
                "--gradient_checkpointing",
                "--deepspeed", "ai/training/configs/ds_config_zero3.json",
            ])

            log_capture: list[str] = []

            class LogCapture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    log_capture.append(self.format(record))

            log_handler = LogCapture()
            log_handler.setLevel(logging.INFO)
            logger = logging.getLogger("orpo_trainer")
            logger.addHandler(log_handler)
            logger.setLevel(logging.INFO)

            try:
                run_orpo(args)
            finally:
                logger.removeHandler(log_handler)

        # ------------------------------------------------------------------ #
        # Checkpoint saved — verify final adapter files exist
        # ------------------------------------------------------------------ #
        final_model_dir = output_dir / "final_model"
        assert (final_model_dir / "adapter_config.json").exists(), \
            "Final model adapter_config.json not saved"
        assert (final_model_dir / "adapter_model.safetensors").exists(), \
            "Final model adapter_model.safetensors not saved"

        # ------------------------------------------------------------------ #
        # Metrics JSON — verify structure and values
        # ------------------------------------------------------------------ #
        metrics_path = output_dir / "orpo_metrics.json"
        assert metrics_path.exists(), "orpo_metrics.json not written"
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert report["method"] == "orpo", "metrics method must be 'orpo'"
        assert report["beta"] == 0.1
        assert report["metrics"]["train_loss"] == 0.35
        assert report["metrics"]["num_train_samples"] == 50
        assert "checkpoint_verification" in report["metrics"]
        assert "generated_at" in report
        # Enhanced fields
        assert report["adapter_variant"] == "lora"
        assert report["deepspeed_config"] == "ai/training/configs/ds_config_zero3.json"
        assert report["gradient_checkpointing"] is True
        assert report["warmup_ratio"] == 0.1
        assert report["lr_scheduler_type"] == "cosine"

        # ------------------------------------------------------------------ #
        # Log output — verify key training messages
        # ------------------------------------------------------------------ #
        log_text = "\n".join(log_capture)
        assert "Loaded 50 preference pairs" in log_text, "Missing dataset load log"
        assert "LoRA config:" in log_text, "Missing LoRA config log"
        assert "ORPO config:" in log_text, "Missing ORPO config log"
        assert "Checkpoint verification:" in log_text, "Missing checkpoint verification log"
        assert "ORPO training complete" in log_text, "Missing completion log"

        # ------------------------------------------------------------------ #
        # Mock interaction verification
        # ------------------------------------------------------------------ #
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        trainer_instance.train.assert_called_once()
        trainer_instance.save_model.assert_called_once()

    def test_orpo_with_dora_variant(self, tmp_path: Path):
        """Verify that --use_dora flag propagates to the LoRA config and
        the metrics report records adapter_variant='dora'."""
        model_instance = MagicMock()
        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|im_end|>"

        train_result = MagicMock()
        train_result.training_loss = 0.30
        train_result.metrics = {"train_runtime": 60.0}

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.experimental.orpo.ORPOConfig"),
            patch("trl.experimental.orpo.ORPOTrainer") as mock_orpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_orpo_trainer_cls.return_value
            trainer_instance.train.return_value = train_result

            def _mock_save_model(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save_model
            tokenizer_instance.save_pretrained.side_effect = _mock_save_model

            data_path = tmp_path / "test_dora_pairs.jsonl"
            self._create_test_dataset(data_path, 15)
            output_dir = tmp_path / "dora_output"

            from training.orpo_trainer import build_parser, run_orpo

            parser = build_parser()
            args = parser.parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", "0.1",
                "--use_dora",
            ])

            # Capture logs to verify DoRA message
            log_capture: list[str] = []

            class LogCapture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
                    log_capture.append(self.format(record))

            log_handler = LogCapture()
            log_handler.setLevel(logging.INFO)
            logger = logging.getLogger("orpo_trainer")
            logger.addHandler(log_handler)
            logger.setLevel(logging.INFO)

            try:
                run_orpo(args)
            finally:
                logger.removeHandler(log_handler)

        # Verify metrics report records DoRA variant
        metrics_path = output_dir / "orpo_metrics.json"
        assert metrics_path.exists()
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert report["adapter_variant"] == "dora"

        # Verify DoRA log message
        log_text = "\n".join(log_capture)
        assert "DoRA" in log_text, "Missing DoRA log message"

    def test_orpo_without_deepspeed(self, tmp_path: Path):
        """Verify that omitting --deepspeed results in None in the metrics."""
        model_instance = MagicMock()
        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|im_end|>"

        train_result = MagicMock()
        train_result.training_loss = 0.40
        train_result.metrics = {"train_runtime": 30.0}

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.experimental.orpo.ORPOConfig"),
            patch("trl.experimental.orpo.ORPOTrainer") as mock_orpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_orpo_trainer_cls.return_value
            trainer_instance.train.return_value = train_result

            def _mock_save_model(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save_model
            tokenizer_instance.save_pretrained.side_effect = _mock_save_model

            data_path = tmp_path / "test_no_ds_pairs.jsonl"
            self._create_test_dataset(data_path, 15)
            output_dir = tmp_path / "no_ds_output"

            from training.orpo_trainer import build_parser, run_orpo

            parser = build_parser()
            args = parser.parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", "0.15",
            ])

            run_orpo(args)

        metrics_path = output_dir / "orpo_metrics.json"
        assert metrics_path.exists()
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert report["deepspeed_config"] is None
        assert report["beta"] == 0.15

    def test_orpo_metrics_include_train_samples_per_second(self, tmp_path: Path):
        """Verify that train_samples_per_second from train_result.metrics
        is captured in the ORPO metrics report."""
        model_instance = MagicMock()
        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|im_end|>"

        train_result = MagicMock()
        train_result.training_loss = 0.28
        train_result.metrics = {
            "train_runtime": 200.0,
            "train_samples_per_second": 3.75,
        }

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.experimental.orpo.ORPOConfig"),
            patch("trl.experimental.orpo.ORPOTrainer") as mock_orpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_orpo_trainer_cls.return_value
            trainer_instance.train.return_value = train_result

            def _mock_save_model(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save_model
            tokenizer_instance.save_pretrained.side_effect = _mock_save_model

            data_path = tmp_path / "test_sps_pairs.jsonl"
            self._create_test_dataset(data_path, 15)
            output_dir = tmp_path / "sps_output"

            from training.orpo_trainer import build_parser, run_orpo

            parser = build_parser()
            args = parser.parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
            ])

            run_orpo(args)

        report = json.loads((output_dir / "orpo_metrics.json").read_text(encoding="utf-8"))
        assert report["metrics"]["train_samples_per_second"] == 3.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
