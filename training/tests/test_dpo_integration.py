"""Integration test for DPO trainer end-to-end with mocked model loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from training.dpo_trainer import build_parser, run_dpo, save_metrics, CheckpointVerificationCallback


class TestDPOIntegration:
    def _create_test_dataset(self, path: Path, num_samples: int = 50) -> None:
        records = [
            {
                "prompt": f"Q{i}",
                "chosen": f"Good answer {i}",
                "rejected": f"Bad answer {i}",
            }
            for i in range(num_samples)
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_dpo_end_to_end_mocked(self, tmp_path: Path):
        model_instance = MagicMock()
        model_instance.device_map = "auto"

        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|endoftext|>"

        train_result = MagicMock()
        train_result.training_loss = 0.5
        train_result.metrics = {"train_runtime": 100.0, "train_samples_per_second": 2.5}

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.DPOConfig") as mock_dpo_config_cls,
            patch("trl.DPOTrainer") as mock_dpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_dpo_trainer_cls.return_value
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
            output_dir = tmp_path / "dpo_output"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--data_path",
                    str(data_path),
                    "--base_model_checkpoint",
                    "mock-model",
                    "--output_dir",
                    str(output_dir),
                    "--beta",
                    "0.1",
                    "--max_seq_length",
                    "512",
                    "--batch_size",
                    "2",
                    "--learning_rate",
                    "5e-5",
                    "--epochs",
                    "1",
                    "--logging_steps",
                    "5",
                ]
            )

            log_capture = []

            class LogCapture(logging.Handler):
                def emit(self, record):
                    log_capture.append(self.format(record))

            log_handler = LogCapture()
            log_handler.setLevel(logging.INFO)
            logger = logging.getLogger("dpo_trainer")
            logger.addHandler(log_handler)
            logger.setLevel(logging.INFO)

            try:
                run_dpo(args)
            finally:
                logger.removeHandler(log_handler)

        # Verify final adapter saved
        final_model_dir = output_dir / "final_model"
        assert (final_model_dir / "adapter_config.json").exists()
        assert (final_model_dir / "adapter_model.safetensors").exists()

        # Verify metrics JSON written
        metrics_path = output_dir / "dpo_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "generated_at" in metrics
        assert metrics["beta"] == 0.1
        assert metrics["metrics"]["train_loss"] == 0.5
        assert "checkpoint_verification" in metrics["metrics"]

        # Verify logs present
        log_text = "\n".join(log_capture)
        assert "Loaded 50 preference pairs" in log_text
        assert "LoRA config:" in log_text
        assert "Checkpoint verification:" in log_text
        assert "DPO training complete" in log_text

        # Verify mocks called
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        trainer_instance.train.assert_called_once()
        trainer_instance.save_model.assert_called_once()

    def test_save_metrics_creates_valid_json(self, tmp_path: Path):
        metrics = {
            "train_loss": 0.42,
            "train_runtime": 150.5,
            "beta": 0.1,
            "checkpoint_verification": {
                "adapter_config.json": True,
                "adapter_model.safetensors": True,
            },
        }

        save_metrics(tmp_path, metrics, beta=0.1)

        metrics_path = tmp_path / "dpo_metrics.json"
        assert metrics_path.exists()
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert report["beta"] == 0.1
        assert report["metrics"]["train_loss"] == 0.42
        assert "generated_at" in report

    def test_checkpoint_verification_callback(self, tmp_path: Path):
        callback = CheckpointVerificationCallback()

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = callback.verify(empty_dir)
        assert result == {"adapter_config.json": False, "adapter_model.safetensors": False}

        full_dir = tmp_path / "full"
        full_dir.mkdir()
        (full_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        (full_dir / "adapter_model.safetensors").write_text("data", encoding="utf-8")

        result = callback.verify(full_dir)
        assert result == {"adapter_config.json": True, "adapter_model.safetensors": True}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
