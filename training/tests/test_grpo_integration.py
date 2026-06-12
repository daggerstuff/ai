"""Integration test for GRPO trainer end-to-end with mocked model loading."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.grpo_trainer import run_grpo, build_parser


class TestGRPOIntegration:
    def _create_test_dataset(self, path: Path, num_samples: int = 50) -> None:
        records = [
            {"prompt": f"User: I'm feeling anxious about my job interview tomorrow. Can you help? {i}"}
            for i in range(num_samples)
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_grpo_end_to_end_mocked(self, tmp_path: Path):
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
            patch("trl.GRPOConfig") as mock_grpo_config_cls,
            patch("trl.GRPOTrainer") as mock_grpo_trainer_cls,
        ):
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = tokenizer_instance

            trainer_instance = mock_grpo_trainer_cls.return_value
            trainer_instance.train.return_value = train_result

            def _mock_save_model(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save_model
            tokenizer_instance.save_pretrained.side_effect = _mock_save_model

            data_path = tmp_path / "test_prompts.jsonl"
            self._create_test_dataset(data_path, 50)
            output_dir = tmp_path / "grpo_output"

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--data_path",
                    str(data_path),
                    "--base_model_checkpoint",
                    "mock-model",
                    "--output_dir",
                    str(output_dir),
                    "--empathy_weight",
                    "0.5",
                    "--crisis_weight",
                    "0.3",
                    "--clinical_validity_weight",
                    "0.2",
                    "--min_reward_threshold",
                    "0.3",
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
            logger = logging.getLogger("grpo_trainer")
            logger.addHandler(log_handler)
            logger.setLevel(logging.INFO)

            try:
                run_grpo(args)
            finally:
                logger.removeHandler(log_handler)

        final_model_dir = output_dir / "final_model"
        assert (final_model_dir / "adapter_config.json").exists()
        assert (final_model_dir / "adapter_model.safetensors").exists()

        metrics_path = output_dir / "grpo_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "generated_at" in metrics
        assert "metrics" in metrics
        assert metrics["metrics"]["train_loss"] == 0.5
        assert metrics["metrics"]["empathy_weight"] == 0.5
        assert metrics["metrics"]["crisis_weight"] == 0.3
        assert metrics["metrics"]["clinical_validity_weight"] == 0.2
        assert metrics["metrics"]["min_reward_threshold"] == 0.3

        log_text = "\n".join(log_capture)
        assert "Reward weights:" in log_text
        assert "GRPO training complete" in log_text

        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        trainer_instance.train.assert_called_once()
        trainer_instance.save_model.assert_called_once()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
