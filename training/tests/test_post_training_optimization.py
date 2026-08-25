"""Tests for post-training optimization scripts (PIX-4345 Appendix E Steps 6-7).

Tests for:
  - prune_adapter.py  (magnitude-based LoRA pruning + quality verification)
  - quantize_model.py (AWQ / GPTQ / GGUF quantization for inference)
  - distill_model.py  (knowledge distillation pipeline)

All heavy ML deps (torch, transformers, peft, autoawq, auto_gptq) are mocked
so tests run without GPU or model downloads.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------


class TestPruningStats:
    def test_defaults(self):
        from training.prune_adapter import PruningStats

        stats = PruningStats(adapter_path="./lora-out", amount=0.3)
        assert stats.total_params == 0
        assert stats.pruned_params == 0
        assert stats.sparsity == 0.0
        assert stats.target_modules == []
        assert stats.quality_gate_passed is None

    def test_to_dict(self):
        from training.prune_adapter import PruningStats

        stats = PruningStats(
            adapter_path="./lora-out",
            amount=0.3,
            total_params=1000,
            pruned_params=300,
            sparsity=0.3,
            target_modules=["q_proj", "v_proj"],
        )
        d = stats.to_dict()
        assert d["total_params"] == 1000
        assert d["pruned_params"] == 300
        assert d["sparsity"] == 0.3
        assert "q_proj" in d["target_modules"]


class TestExtractDomainScore:
    def test_found(self):
        from training.prune_adapter import _extract_domain_score

        report = {
            "results": [
                {"name": "mmlu", "score": 0.72, "category": "general"},
                {"name": "domain_clinical_empathy", "score": 0.48, "category": "domain"},
            ]
        }
        assert _extract_domain_score(report) == 0.48

    def test_not_found(self):
        from training.prune_adapter import _extract_domain_score

        report = {"results": [{"name": "mmlu", "score": 0.72, "category": "general"}]}
        assert _extract_domain_score(report) is None

    def test_empty(self):
        from training.prune_adapter import _extract_domain_score

        assert _extract_domain_score({}) is None
        assert _extract_domain_score({"results": []}) is None


class TestVerifyQuality:
    def test_pass(self, tmp_path):
        from training.prune_adapter import verify_quality

        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": 0.50, "category": "domain"}]}
            )
        )
        post_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": 0.49, "category": "domain"}]}
            )
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is True
        assert result["quality_loss"] == pytest.approx(0.02, abs=0.01)
        assert "PASS" in result["verdict"]

    def test_fail(self, tmp_path):
        from training.prune_adapter import verify_quality

        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": 0.50, "category": "domain"}]}
            )
        )
        post_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": 0.40, "category": "domain"}]}
            )
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is False
        assert result["quality_loss"] == pytest.approx(0.20, abs=0.01)
        assert "FAIL" in result["verdict"]

    def test_missing_domain(self, tmp_path):
        from training.prune_adapter import verify_quality

        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps({"results": [{"name": "mmlu", "score": 0.72, "category": "general"}]})
        )
        post_report.write_text(
            json.dumps({"results": [{"name": "mmlu", "score": 0.71, "category": "general"}]})
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is None
        assert "INSUFFICIENT_DATA" in result["verdict"]


class TestPruneLoraAdapter:
    def test_no_lora_modules(self):
        from training.prune_adapter import prune_lora_adapter

        model = MagicMock()
        model.named_modules.return_value = []
        stats = prune_lora_adapter(model, amount=0.3)
        assert stats.total_params == 0
        assert stats.pruned_params == 0
        assert stats.sparsity == 0.0

    def test_with_real_linear(self):
        """Test pruning with a real nn.Linear and torch.nn.utils.prune."""
        from training.prune_adapter import prune_lora_adapter

        import torch
        import torch.nn as nn

        # Create a real linear layer with random weights.
        linear = nn.Linear(8, 8)
        # Wrap in a simple module that looks like a LoRA layer.
        class _MockLoraModule(nn.Module):
            def __init__(self, lin: nn.Linear) -> None:
                super().__init__()
                self.lora_A = nn.ModuleDict({"default": lin})
                self.lora_B = nn.ModuleDict({"default": lin})

        lora_mod = _MockLoraModule(linear)

        class _MockModel(nn.Module):
            def __init__(self, mod: nn.Module) -> None:
                super().__init__()
                self.layer = mod

        model = _MockModel(lora_mod)

        stats = prune_lora_adapter(model, amount=0.3)
        assert stats.total_params > 0
        assert stats.pruned_params > 0
        assert stats.sparsity > 0.0
        assert len(stats.target_modules) > 0


class TestPruneCLI:
    def test_dry_run(self, tmp_path):
        from training.prune_adapter import main

        adapter_path = tmp_path / "lora-out"
        adapter_path.mkdir()
        out_dir = tmp_path / "pruning"

        with patch("sys.argv", [
            "prune_adapter",
            "--adapter-path", str(adapter_path),
            "--amount", "0.3",
            "--dry-run",
            "--out-dir", str(out_dir),
        ]):
            ret = main()
        assert ret == 0

        report_files = list(out_dir.glob("prune_plan_*.json"))
        assert len(report_files) == 1
        data = json.loads(report_files[0].read_text())
        assert data["amount"] == 0.3

    def test_missing_base_model_error(self, tmp_path):
        from training.prune_adapter import main

        adapter_path = tmp_path / "lora-out"
        adapter_path.mkdir()

        with patch("sys.argv", [
            "prune_adapter",
            "--adapter-path", str(adapter_path),
            "--amount", "0.3",
            "--out-dir", str(tmp_path / "pruning"),
        ]):
            ret = main()
        assert ret == 1


# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------


class TestQuantizationStats:
    def test_defaults(self):
        from training.quantize_model import QuantizationStats

        stats = QuantizationStats(model_path="./model", method="awq")
        assert stats.bits is None
        assert stats.gguf_quant is None
        assert stats.compression_ratio == 0.0
        assert stats.quality_gate_passed is None

    def test_awq_stats(self):
        from training.quantize_model import QuantizationStats

        stats = QuantizationStats(
            model_path="./model",
            method="awq",
            bits=4,
            original_size_mb=14000.0,
            quantized_size_mb=3500.0,
            compression_ratio=4.0,
        )
        d = stats.to_dict()
        assert d["bits"] == 4
        assert d["compression_ratio"] == 4.0

    def test_gguf_stats(self):
        from training.quantize_model import QuantizationStats

        stats = QuantizationStats(
            model_path="./model",
            method="gguf",
            gguf_quant="q4_k_m",
            quantized_size_mb=4000.0,
        )
        d = stats.to_dict()
        assert d["gguf_quant"] == "q4_k_m"
        assert d["bits"] is None


class TestDirSizeMB:
    def test_single_file(self, tmp_path):
        from training.quantize_model import _dir_size_mb

        f = tmp_path / "test.bin"
        f.write_bytes(b"x" * (2 * 1024 * 1024))  # 2 MB
        size = _dir_size_mb(f)
        assert size == pytest.approx(2.0, abs=0.01)

    def test_directory(self, tmp_path):
        from training.quantize_model import _dir_size_mb

        (tmp_path / "a.bin").write_bytes(b"x" * (1024 * 1024))
        (tmp_path / "b.bin").write_bytes(b"x" * (1024 * 1024))
        size = _dir_size_mb(tmp_path)
        assert size == pytest.approx(2.0, abs=0.01)

    def test_empty_dir(self, tmp_path):
        from training.quantize_model import _dir_size_mb

        assert _dir_size_mb(tmp_path) == 0.0


class TestQuantizeCLI:
    def test_dry_run_awq(self, tmp_path):
        from training.quantize_model import main

        model_path = tmp_path / "model"
        model_path.mkdir()
        out_dir = tmp_path / "quantized"

        with patch("sys.argv", [
            "quantize_model",
            "--model-path", str(model_path),
            "--method", "awq",
            "--bits", "4",
            "--dry-run",
            "--out-dir", str(out_dir),
        ]):
            ret = main()
        assert ret == 0

        plan = json.loads((out_dir / "quantize_plan.json").read_text())
        assert plan["method"] == "awq"
        assert plan["bits"] == 4

    def test_dry_run_gguf(self, tmp_path):
        from training.quantize_model import main

        model_path = tmp_path / "model"
        model_path.mkdir()
        out_dir = tmp_path / "quantized"

        with patch("sys.argv", [
            "quantize_model",
            "--model-path", str(model_path),
            "--method", "gguf",
            "--gguf-quant", "q5_k_m",
            "--dry-run",
            "--out-dir", str(out_dir),
        ]):
            ret = main()
        assert ret == 0

        plan = json.loads((out_dir / "quantize_plan.json").read_text())
        assert plan["method"] == "gguf"
        assert plan["gguf_quant"] == "q5_k_m"
        assert plan["bits"] is None


class TestSupportedMethods:
    def test_methods_exist(self):
        from training.quantize_model import SUPPORTED_METHODS

        assert "awq" in SUPPORTED_METHODS
        assert "gptq" in SUPPORTED_METHODS
        assert "gguf" in SUPPORTED_METHODS

    def test_gguf_quants_exist(self):
        from training.quantize_model import GGUF_QUANTS

        assert "q4_k_m" in GGUF_QUANTS
        assert "q8_0" in GGUF_QUANTS
        assert "f16" in GGUF_QUANTS


# ---------------------------------------------------------------------------
# Distillation tests
# ---------------------------------------------------------------------------


class TestDistillationStats:
    def test_defaults(self):
        from training.distill_model import DistillationStats

        stats = DistillationStats(
            teacher_model="Qwen/Qwen2.5-72B",
            student_model="Qwen/Qwen2.5-7B",
        )
        assert stats.kd_alpha == 0.5
        assert stats.kd_beta == 0.5
        assert stats.temperature == 2.0
        assert stats.epochs == 0
        assert stats.final_loss is None

    def test_to_dict(self):
        from training.distill_model import DistillationStats

        stats = DistillationStats(
            teacher_model="teacher",
            student_model="student",
            epochs=3,
            train_samples=10000,
            final_loss=1.23,
        )
        d = stats.to_dict()
        assert d["epochs"] == 3
        assert d["train_samples"] == 10000
        assert d["final_loss"] == 1.23


class TestKDLoss:
    def test_returns_three_values(self):
        """Test that kd_loss returns (total, kd_loss, ce_loss) tuple."""
        from training.distill_model import kd_loss

        # Mock torch and torch.nn.functional.
        import torch
        import torch.nn.functional as F

        batch, seq_len, vocab = 2, 10, 100
        student_logits = torch.randn(batch, seq_len, vocab)
        teacher_logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))

        total, kd, ce = kd_loss(student_logits, teacher_logits, labels)
        assert total is not None
        assert kd is not None
        assert ce is not None
        # Total should be alpha * kd + beta * ce.
        assert torch.isfinite(total)

    def test_alpha_beta_weights(self):
        """Test that alpha and beta correctly weight the loss terms."""
        from training.distill_model import kd_loss

        import torch

        batch, seq_len, vocab = 1, 5, 50
        student_logits = torch.randn(batch, seq_len, vocab)
        teacher_logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))

        # With alpha=1, beta=0: total should equal kd loss.
        total_kd_only, kd, ce = kd_loss(
            student_logits, teacher_logits, labels, alpha=1.0, beta=0.0
        )
        assert torch.allclose(total_kd_only, kd, atol=1e-5)

        # With alpha=0, beta=1: total should equal ce loss.
        total_ce_only, kd, ce = kd_loss(
            student_logits, teacher_logits, labels, alpha=0.0, beta=1.0
        )
        assert torch.allclose(total_ce_only, ce, atol=1e-5)


class TestDistillCLI:
    def test_train_dry_run(self, tmp_path):
        from training.distill_model import main

        teacher_outputs = tmp_path / "teacher_outputs.jsonl"
        teacher_outputs.write_text('{"messages": []}\n')
        out_dir = tmp_path / "distill-out"

        with patch("sys.argv", [
            "distill_model",
            "train",
            "--teacher-outputs", str(teacher_outputs),
            "--student-model", "Qwen/Qwen2.5-7B",
            "--out-dir", str(out_dir),
            "--dry-run",
        ]):
            ret = main()
        assert ret == 0

        plan = json.loads((out_dir / "distill_plan.json").read_text())
        assert plan["student_model"] == "Qwen/Qwen2.5-7B"
        assert plan["kd_alpha"] == 0.5
        assert plan["kd_beta"] == 0.5

    def test_missing_command(self):
        from training.distill_model import main

        with patch("sys.argv", ["distill_model"]):
            with pytest.raises(SystemExit):
                main()
