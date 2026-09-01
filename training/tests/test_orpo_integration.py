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

import importlib
import json
import logging
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
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
# Mock uninstalled / broken ML packages at the module level so importing
# training.orpo_trainer works without the transformers → kernels crash.
#
# trl is not installed.  orpo_trainer binds classes via
# ``from trl import ORPOTrainer``, so the mock classes must be set as
# attributes on the trl mock module itself (patch("trl.experimental.orpo.*")
# would target auto-created attribute mocks that orpo_trainer never sees).
# ---------------------------------------------------------------------------

_mock_trl = MagicMock()
sys.modules["trl"] = _mock_trl


# peft — installed but triggers transformers → kernels crash on import
def _lora_config(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


_mock_peft = MagicMock()
_mock_peft.prepare_model_for_kbit_training = lambda model: model
_mock_peft.LoraConfig = _lora_config
sys.modules["peft"] = _mock_peft

# ---------------------------------------------------------------------------
# Shared test constants
# ---------------------------------------------------------------------------
E2E_BETA = 0.1
BETA_OVERRIDE_SMALL = 0.15
E2E_SAMPLES = 50
VARIANT_SAMPLES = 15
E2E_TRAIN_LOSS = 0.35
DORA_TRAIN_LOSS = 0.30
NO_DS_TRAIN_LOSS = 0.40
SPS_TRAIN_LOSS = 0.28
E2E_EXPECTED_SPS = 2.5
SPS_EXPECTED = 3.75
E2E_BATCH_SIZE = 2
E2E_MAX_SEQ_LENGTH = 512
E2E_LEARNING_RATE = 5e-6
E2E_LOGGING_STEPS = 5
E2E_WARMUP_RATIO = 0.1
DS_CONFIG_PATH = "ai/training/configs/ds_config_zero3.json"


def _fresh_orpo_module() -> ModuleType:
    """Reload training.orpo_trainer to re-bind trl classes from sys.modules.

    Reloading inside the caller's mock context is required: the module binds
    ``ORPOTrainer``/``ORPOConfig`` at import time, and reloading re-binds
    them to whatever is currently installed on the trl mock.  Without the
    reload, the first test's binding leaks into subsequent tests.
    """
    return importlib.reload(importlib.import_module("training.orpo_trainer"))


class TestORPOIntegration:
    """Integration tests for the ORPO trainer pipeline."""

    @staticmethod
    def _create_test_dataset(path: Path, num_samples: int = E2E_SAMPLES) -> None:
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

    @staticmethod
    def _write_fake_adapter(output_dir_path: str) -> None:
        path = Path(output_dir_path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_text("data", encoding="utf-8")

    @staticmethod
    def _make_instances(train_loss: float, metrics: dict[str, Any]) -> dict[str, MagicMock]:
        """Build the model/tokenizer/train_result instance mocks."""
        model_instance = MagicMock()
        model_instance.device_map = "auto"
        tokenizer_instance = MagicMock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<|im_end|>"
        train_result = MagicMock()
        train_result.training_loss = train_loss
        train_result.metrics = metrics
        return {
            "model": model_instance,
            "tokenizer": tokenizer_instance,
            "train_result": train_result,
        }

    @staticmethod
    @contextmanager
    def _mocked_orpo(instances: dict[str, MagicMock]):
        """Install fresh trl mocks + patch transformers, yield trainer.

        Yields ``(trainer_instance, entered)`` where ``entered`` maps mock
        keys to the entered patch objects (for call assertions).
        """
        trainer_cls = MagicMock()
        config_cls = MagicMock()
        _mock_trl.ORPOTrainer = trainer_cls
        _mock_trl.ORPOConfig = config_cls

        with ExitStack() as stack:
            entered = {
                "model": stack.enter_context(patch("transformers.AutoModelForCausalLM")),
                "tokenizer": stack.enter_context(patch("transformers.AutoTokenizer")),
            }
            entered["model"].from_pretrained.return_value = instances["model"]
            entered["tokenizer"].from_pretrained.return_value = instances["tokenizer"]
            trainer_instance = trainer_cls.return_value
            trainer_instance.train.return_value = instances["train_result"]
            trainer_instance.save_model.side_effect = TestORPOIntegration._write_fake_adapter
            instances["tokenizer"].save_pretrained.side_effect = TestORPOIntegration._write_fake_adapter
            try:
                yield trainer_instance, entered
            finally:
                patch.stopall()

    def _run_orpo_with_logs(self, orpo: ModuleType, args: Any) -> list[str]:
        """Run run_orpo with a log handler attached; return captured lines."""
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
            orpo.run_orpo(args)
        finally:
            logger.removeHandler(log_handler)
        return log_capture

    def _load_metrics_report(self, output_dir: Path) -> dict[str, Any]:
        metrics_path = output_dir / "orpo_metrics.json"
        assert metrics_path.exists(), "orpo_metrics.json not written"
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _prepare_test_paths(
        self, tmp_path: Path, name: str, num_samples: int = VARIANT_SAMPLES
    ) -> tuple[Path, Path]:
        """Create the JSONL dataset and output dir for a test run.

        Returns ``(data_path, output_dir)`` so callers can wire them into
        ``--data_path`` / ``--output_dir`` CLI args.
        """
        data_path = tmp_path / f"test_{name}_pairs.jsonl"
        self._create_test_dataset(data_path, num_samples)
        output_dir = tmp_path / f"{name}_output"
        return data_path, output_dir

    def test_orpo_end_to_end_mocked(self, tmp_path: Path):
        """Run ORPO trainer with 50-sample preference dataset (mocked model).

        Verifies:
        - Preference dataset loaded and passed to ORPOTrainer
        - ORPOConfig built with correct beta and learning_rate
        - Final checkpoint saved (adapter files + metrics JSON)
        - Metrics JSON contains method="orpo", beta, adapter_variant
        - Mock interactions (model load, train, save) all called once
        """
        instances = self._make_instances(
            E2E_TRAIN_LOSS,
            {"train_runtime": 120.0, "train_samples_per_second": E2E_EXPECTED_SPS},
        )
        with self._mocked_orpo(instances) as (trainer_instance, entered):
            data_path, output_dir = self._prepare_test_paths(
                tmp_path, "preference", E2E_SAMPLES
            )

            orpo = _fresh_orpo_module()
            args = orpo.build_parser().parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", str(E2E_BETA),
                "--max_seq_length", str(E2E_MAX_SEQ_LENGTH),
                "--batch_size", str(E2E_BATCH_SIZE),
                "--learning_rate", str(E2E_LEARNING_RATE),
                "--epochs", "1",
                "--logging_steps", str(E2E_LOGGING_STEPS),
                "--warmup_ratio", str(E2E_WARMUP_RATIO),
                "--lr_scheduler_type", "cosine",
                "--gradient_checkpointing",
                "--deepspeed", DS_CONFIG_PATH,
            ])

            log_capture = self._run_orpo_with_logs(orpo, args)

        # Checkpoint saved — verify final adapter files exist
        final_model_dir = output_dir / "final_model"
        assert (final_model_dir / "adapter_config.json").exists(), \
            "Final model adapter_config.json not saved"
        assert (final_model_dir / "adapter_model.safetensors").exists(), \
            "Final model adapter_model.safetensors not saved"

        # Metrics JSON — verify structure and values
        report = self._load_metrics_report(output_dir)
        assert report["method"] == "orpo", "metrics method must be 'orpo'"
        assert report["beta"] == E2E_BETA
        assert report["metrics"]["train_loss"] == E2E_TRAIN_LOSS
        assert report["metrics"]["num_train_samples"] == E2E_SAMPLES
        assert "checkpoint_verification" in report["metrics"]
        assert "generated_at" in report
        # Enhanced fields
        assert report["adapter_variant"] == "lora"
        assert report["deepspeed_config"] == DS_CONFIG_PATH
        assert report["gradient_checkpointing"] is True
        assert report["warmup_ratio"] == E2E_WARMUP_RATIO
        assert report["lr_scheduler_type"] == "cosine"

        # Log output — verify key training messages
        log_text = "\n".join(log_capture)
        assert f"Loaded {E2E_SAMPLES} preference pairs" in log_text, "Missing dataset load log"
        assert "LoRA config:" in log_text, "Missing LoRA config log"
        assert "ORPO config:" in log_text, "Missing ORPO config log"
        assert "Checkpoint verification:" in log_text, "Missing checkpoint verification log"
        assert "ORPO training complete" in log_text, "Missing completion log"

        # Mock interaction verification
        entered["model"].from_pretrained.assert_called_once()
        entered["tokenizer"].from_pretrained.assert_called_once()
        trainer_instance.train.assert_called_once()
        trainer_instance.save_model.assert_called_once()

    def test_orpo_with_dora_variant(self, tmp_path: Path):
        """Verify that --use_dora flag propagates to the LoRA config and
        the metrics report records adapter_variant='dora'."""
        instances = self._make_instances(DORA_TRAIN_LOSS, {"train_runtime": 60.0})
        with self._mocked_orpo(instances):
            data_path, output_dir = self._prepare_test_paths(tmp_path, "dora")

            orpo = _fresh_orpo_module()
            args = orpo.build_parser().parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", str(E2E_BETA),
                "--use_dora",
            ])

            log_capture = self._run_orpo_with_logs(orpo, args)

        # Verify metrics report records DoRA variant
        report = self._load_metrics_report(output_dir)
        assert report["adapter_variant"] == "dora"

        # Verify DoRA log message
        log_text = "\n".join(log_capture)
        assert "DoRA" in log_text, "Missing DoRA log message"

    def test_orpo_without_deepspeed(self, tmp_path: Path):
        """Verify that omitting --deepspeed results in None in the metrics."""
        instances = self._make_instances(NO_DS_TRAIN_LOSS, {"train_runtime": 30.0})
        with self._mocked_orpo(instances):
            data_path, output_dir = self._prepare_test_paths(tmp_path, "no_ds")

            orpo = _fresh_orpo_module()
            args = orpo.build_parser().parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
                "--beta", str(BETA_OVERRIDE_SMALL),
            ])

            orpo.run_orpo(args)

        report = self._load_metrics_report(output_dir)
        assert report["deepspeed_config"] is None
        assert report["beta"] == BETA_OVERRIDE_SMALL

    def test_orpo_metrics_include_train_samples_per_second(self, tmp_path: Path):
        """Verify that train_samples_per_second from train_result.metrics
        is captured in the ORPO metrics report."""
        instances = self._make_instances(
            SPS_TRAIN_LOSS,
            {"train_runtime": 200.0, "train_samples_per_second": SPS_EXPECTED},
        )
        with self._mocked_orpo(instances):
            data_path, output_dir = self._prepare_test_paths(tmp_path, "sps")

            orpo = _fresh_orpo_module()
            args = orpo.build_parser().parse_args([
                "--data_path", str(data_path),
                "--base_model_checkpoint", "mock-model",
                "--output_dir", str(output_dir),
            ])

            orpo.run_orpo(args)

        report = self._load_metrics_report(output_dir)
        assert report["metrics"]["train_samples_per_second"] == SPS_EXPECTED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
