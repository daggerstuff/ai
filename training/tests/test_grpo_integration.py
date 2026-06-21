"""Integration test for GRPO trainer end-to-end with mocked model loading.

Validates:
1. Reward scores logged with correct weight values
2. Low-reward filtering integration (reward_funcs passed to GRPOTrainer)
3. Checkpoint saved (final adapter files and metrics JSON)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
# patch("trl.*") and the lazy imports inside run_grpo() work without
# triggering the transformers -> kernels import-time crash.
# ---------------------------------------------------------------------------

# trl — GRPOConfig is lazy-loaded and may trigger transformers/kernels crash
_mock_trl = MagicMock()
_mock_trl.GRPOTrainer = MagicMock()
_mock_trl.GRPOConfig = MagicMock()
sys.modules["trl"] = _mock_trl


# peft — installed but triggers transformers -> kernels crash on import
def _lora_config(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


_mock_peft = MagicMock()
_mock_peft.prepare_model_for_kbit_training = lambda model: model
_mock_peft.LoraConfig = _lora_config
sys.modules["peft"] = _mock_peft


class TestGRPOIntegration:
    """Integration tests for the GRPO trainer pipeline."""

    def _create_test_dataset(self, path: Path, num_samples: int = 50) -> None:
        """Write a JSONL prompt dataset for testing."""
        records = [
            {"prompt": f"User: I'm feeling anxious about my job interview tomorrow. Can you help? {i}"}
            for i in range(num_samples)
        ]
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    def test_grpo_end_to_end_mocked(self, tmp_path: Path):
        """Run GRPO trainer with 50-sample prompt dataset (mocked model loading).

        Verifies:
        - Reward scores logged with correct weight values
        - Reward function wired into GRPOTrainer (low-reward filtering integration)
        - Final checkpoint saved (adapter files + metrics JSON)
        """
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
            patch("trl.GRPOConfig"),
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

            from training.grpo_trainer import build_parser, run_grpo

            parser = build_parser()
            expected_empathy_weight = 0.5
            expected_crisis_weight = 0.3
            expected_clinical_validity_weight = 0.2
            expected_min_reward_threshold = 0.3

            args = parser.parse_args(
                [
                    "--data_path",
                    str(data_path),
                    "--base_model_checkpoint",
                    "mock-model",
                    "--output_dir",
                    str(output_dir),
                    "--empathy_weight",
                    str(expected_empathy_weight),
                    "--crisis_weight",
                    str(expected_crisis_weight),
                    "--clinical_validity_weight",
                    str(expected_clinical_validity_weight),
                    "--min_reward_threshold",
                    str(expected_min_reward_threshold),
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

            log_capture: list[str] = []

            class LogCapture(logging.Handler):
                def emit(self, record: logging.LogRecord) -> None:
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

        # ------------------------------------------------------------------ #
        # Checkpoint saved — verify final adapter files exist
        # ------------------------------------------------------------------ #
        final_model_dir = output_dir / "final_model"
        assert (final_model_dir / "adapter_config.json").exists(), "Final model adapter_config.json not saved"
        assert (final_model_dir / "adapter_model.safetensors").exists(), (
            "Final model adapter_model.safetensors not saved"
        )

        # ------------------------------------------------------------------ #
        # Metrics JSON — verify structure and values
        # ------------------------------------------------------------------ #
        metrics_path = output_dir / "grpo_metrics.json"
        assert metrics_path.exists(), "grpo_metrics.json not written"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "generated_at" in metrics, "metrics missing generated_at timestamp"
        assert "metrics" in metrics, "metrics missing 'metrics' key"
        assert metrics["metrics"]["train_loss"] == 0.5
        assert metrics["metrics"]["empathy_weight"] == expected_empathy_weight
        assert metrics["metrics"]["crisis_weight"] == expected_crisis_weight
        assert metrics["metrics"]["clinical_validity_weight"] == expected_clinical_validity_weight
        assert metrics["metrics"]["min_reward_threshold"] == expected_min_reward_threshold

        # ------------------------------------------------------------------ #
        # Reward scores logged — verify weight values in log output
        # ------------------------------------------------------------------ #
        log_text = "\n".join(log_capture)
        assert "Reward weights:" in log_text, "Missing reward weights log"
        # Verify the actual weight values are logged
        assert f"empathy={expected_empathy_weight:.2f}" in log_text, (
            f"Expected empathy_weight={expected_empathy_weight:.2f} in log"
        )
        assert f"crisis={expected_crisis_weight:.2f}" in log_text, (
            f"Expected crisis_weight={expected_crisis_weight:.2f} in log"
        )
        assert f"clinical={expected_clinical_validity_weight:.2f}" in log_text, (
            f"Expected clinical_validity_weight={expected_clinical_validity_weight:.2f} in log"
        )
        assert f"threshold={expected_min_reward_threshold:.2f}" in log_text, (
            f"Expected min_reward_threshold={expected_min_reward_threshold:.2f} in log"
        )
        assert "GRPO training complete" in log_text, "Missing completion log"

        # ------------------------------------------------------------------ #
        # Low-reward filtering integration — verify reward_funcs passed
        # ------------------------------------------------------------------ #
        # GRPOTrainer constructor should receive reward_funcs callable
        _, grpo_kwargs = mock_grpo_trainer_cls.call_args
        assert "reward_funcs" in grpo_kwargs, (
            "GRPOTrainer not called with reward_funcs (low-reward filtering integration)"
        )
        assert callable(grpo_kwargs["reward_funcs"]), "reward_funcs must be callable"

        # ------------------------------------------------------------------ #
        # Mock interaction verification
        # ------------------------------------------------------------------ #
        mock_model_cls.from_pretrained.assert_called_once()
        mock_tokenizer_cls.from_pretrained.assert_called_once()
        trainer_instance.train.assert_called_once()
        trainer_instance.save_model.assert_called_once()

    def test_save_metrics_creates_valid_json(self, tmp_path: Path) -> None:
        """Verify that the metrics JSON written by run_grpo has valid structure.

        Uses a mocked training run to exercise the metrics-writing logic.
        """
        from training.grpo_trainer import build_parser, run_grpo

        data_path = tmp_path / "test_prompts.jsonl"
        data_path.write_text('{"prompt": "test"}\n', encoding="utf-8")

        output_dir = tmp_path / "metrics_test_output"

        with (
            patch("transformers.AutoModelForCausalLM") as mock_model_cls,
            patch("transformers.AutoTokenizer") as mock_tokenizer_cls,
            patch("trl.GRPOConfig"),
            patch("trl.GRPOTrainer") as mock_grpo_trainer_cls,
        ):
            model_instance = MagicMock()
            model_instance.device_map = "auto"
            mock_model_cls.from_pretrained.return_value = model_instance
            mock_tokenizer_cls.from_pretrained.return_value = MagicMock()

            trainer_instance = mock_grpo_trainer_cls.return_value
            train_result = MagicMock()
            train_result.training_loss = 0.42
            train_result.metrics = {"train_runtime": 150.5}
            trainer_instance.train.return_value = train_result

            def _mock_save(output_dir_path: str) -> None:
                path = Path(output_dir_path)
                path.mkdir(parents=True, exist_ok=True)
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

            trainer_instance.save_model.side_effect = _mock_save

            parser = build_parser()
            args = parser.parse_args(
                [
                    "--data_path",
                    str(data_path),
                    "--base_model_checkpoint",
                    "mock-model",
                    "--output_dir",
                    str(output_dir),
                ]
            )
            run_grpo(args)

        metrics_path = output_dir / "grpo_metrics.json"
        assert metrics_path.exists(), "grpo_metrics.json not written"
        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "generated_at" in report, "Missing generated_at"
        assert "metrics" in report, "Missing metrics key"
        assert report["metrics"]["train_loss"] == 0.42


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
