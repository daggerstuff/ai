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
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from training.distill_model import DistillationStats, kd_loss, main as distill_main
from training.prune_adapter import (
    PruningStats,
    _extract_domain_score,
    main as prune_main,
    prune_lora_adapter,
    verify_quality,
)
from training.quantize_model import (
    GGUF_QUANTS,
    SUPPORTED_METHODS,
    QuantizationStats,
    _dir_size_mb,
    main as quantize_main,
)

# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------

PRUNE_AMOUNT = 0.3
PRUNED_RATIO = 0.3
TOTAL_PARAMS = 1000
PRUNED_PARAMS = 300
EXPECTED_SCORE = 0.48
GENERAL_SCORE = 0.72
DOMAIN_SCORE_PRE = 0.50
DOMAIN_SCORE_POST_PASS = 0.49
DOMAIN_SCORE_POST_FAIL = 0.40
EXPECTED_QUALITY_LOSS_PASS = 0.02
EXPECTED_QUALITY_LOSS_FAIL = 0.20
APPROX_TOL = 0.01
MB_FILE_SIZE = 2 * 1024 * 1024
MB_SINGLE = 1024 * 1024
BITS_AWQ = 4
COMPRESSION_AWQ = 4.0
KD_ALPHA_DEFAULT = 0.5
KD_BETA_DEFAULT = 0.5
KD_TEMPERATURE_DEFAULT = 2.0
KD_EPOCHS = 3
KD_TRAIN_SAMPLES = 10000
KD_FINAL_LOSS = 1.23
BATCH_SMALL = 2
SEQ_LEN = 10
VOCAB_SMALL = 100
BATCH_SINGLE = 1
SEQ_LEN_SHORT = 5
VOCAB_TINY = 50
ATOL_LOSS = 1e-5
PLAN_AMOUNT_KEY = "amount"

# ---------------------------------------------------------------------------
# Pruning tests
# ---------------------------------------------------------------------------

class TestPruningStats:
    def test_defaults(self):
        stats = PruningStats(adapter_path="./lora-out", amount=PRUNE_AMOUNT)
        assert stats.total_params == 0
        assert stats.pruned_params == 0
        assert stats.sparsity == 0.0
        assert stats.target_modules == []
        assert stats.quality_gate_passed is None

    def test_to_dict(self):
        stats = PruningStats(
            adapter_path="./lora-out",
            amount=PRUNE_AMOUNT,
            total_params=TOTAL_PARAMS,
            pruned_params=PRUNED_PARAMS,
            sparsity=PRUNED_RATIO,
            target_modules=["q_proj", "v_proj"],
        )
        d = stats.to_dict()
        assert d["total_params"] == TOTAL_PARAMS
        assert d["pruned_params"] == PRUNED_PARAMS
        assert d["sparsity"] == PRUNED_RATIO
        assert "q_proj" in d["target_modules"]

class TestExtractDomainScore:
    def test_found(self):
        report = {
            "results": [
                {"name": "mmlu", "score": GENERAL_SCORE, "category": "general"},
                {"name": "domain_clinical_empathy", "score": EXPECTED_SCORE, "category": "domain"},
            ]
        }
        assert _extract_domain_score(report) == EXPECTED_SCORE

    def test_not_found(self):
        report = {"results": [{"name": "mmlu", "score": GENERAL_SCORE, "category": "general"}]}
        assert _extract_domain_score(report) is None

    def test_empty(self):
        assert _extract_domain_score({}) is None
        assert _extract_domain_score({"results": []}) is None

class TestVerifyQuality:
    def test_pass(self, tmp_path):
        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": DOMAIN_SCORE_PRE, "category": "domain"}]}
            )
        )
        post_report.write_text(
            json.dumps(
                {
                    "results": [
                        {"name": "domain_clinical_empathy", "score": DOMAIN_SCORE_POST_PASS, "category": "domain"}
                    ]
                }
            )
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is True
        assert result["quality_loss"] == pytest.approx(EXPECTED_QUALITY_LOSS_PASS, abs=APPROX_TOL)
        assert "PASS" in result["verdict"]

    def test_fail(self, tmp_path):
        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps(
                {"results": [{"name": "domain_clinical_empathy", "score": DOMAIN_SCORE_PRE, "category": "domain"}]}
            )
        )
        post_report.write_text(
            json.dumps(
                {
                    "results": [
                        {"name": "domain_clinical_empathy", "score": DOMAIN_SCORE_POST_FAIL, "category": "domain"}
                    ]
                }
            )
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is False
        assert result["quality_loss"] == pytest.approx(EXPECTED_QUALITY_LOSS_FAIL, abs=APPROX_TOL)
        assert "FAIL" in result["verdict"]

    def test_missing_domain(self, tmp_path):
        pre_report = tmp_path / "pre.json"
        post_report = tmp_path / "post.json"
        pre_report.write_text(
            json.dumps({"results": [{"name": "mmlu", "score": GENERAL_SCORE, "category": "general"}]})
        )
        post_report.write_text(
            json.dumps({"results": [{"name": "mmlu", "score": GENERAL_SCORE - 0.01, "category": "general"}]})
        )

        result = verify_quality(pre_report, post_report)
        assert result["quality_gate_passed"] is None
        assert "INSUFFICIENT_DATA" in result["verdict"]

class TestPruneLoraAdapter:
    def test_no_lora_modules(self):
        model = MagicMock()
        model.named_modules.return_value = []
        stats = prune_lora_adapter(model, amount=PRUNE_AMOUNT)
        assert stats.total_params == 0
        assert stats.pruned_params == 0
        assert stats.sparsity == 0.0

    def test_with_real_linear(self):
        """Test pruning with a real nn.Linear and torch.nn.utils.prune."""
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

        stats = prune_lora_adapter(model, amount=PRUNE_AMOUNT)
        assert stats.total_params > 0
        assert stats.pruned_params > 0
        assert stats.sparsity > 0.0
        assert len(stats.target_modules) > 0

class TestPruneCLI:
    def test_dry_run(self, tmp_path):
        adapter_path = tmp_path / "lora-out"
        adapter_path.mkdir()
        out_dir = tmp_path / "pruning"

        with patch("sys.argv", [
            "prune_adapter",
            "--adapter-path", str(adapter_path),
            "--amount", str(PRUNE_AMOUNT),
            "--dry-run",
            "--out-dir", str(out_dir),
        ]):
            ret = prune_main()
        assert ret == 0

        report_files = list(out_dir.glob("prune_plan_*.json"))
        assert len(report_files) == 1
        data = json.loads(report_files[0].read_text())
        assert data[PLAN_AMOUNT_KEY] == PRUNE_AMOUNT

    def test_missing_base_model_error(self, tmp_path):
        adapter_path = tmp_path / "lora-out"
        adapter_path.mkdir()

        with patch("sys.argv", [
            "prune_adapter",
            "--adapter-path", str(adapter_path),
            "--amount", str(PRUNE_AMOUNT),
            "--out-dir", str(tmp_path / "pruning"),
        ]):
            ret = prune_main()
        assert ret == 1

# ---------------------------------------------------------------------------
# Quantization tests
# ---------------------------------------------------------------------------

class TestQuantizationStats:
    def test_defaults(self):
        stats = QuantizationStats(model_path="./model", method="awq")
        assert stats.bits is None
        assert stats.gguf_quant is None
        assert stats.compression_ratio == 0.0
        assert stats.quality_gate_passed is None

    def test_awq_stats(self):
        stats = QuantizationStats(
            model_path="./model",
            method="awq",
            bits=BITS_AWQ,
            original_size_mb=14000.0,
            quantized_size_mb=3500.0,
            compression_ratio=COMPRESSION_AWQ,
        )
        d = stats.to_dict()
        assert d["bits"] == BITS_AWQ
        assert d["compression_ratio"] == COMPRESSION_AWQ

    def test_gguf_stats(self):
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
        f = tmp_path / "test.bin"
        f.write_bytes(b"x" * MB_FILE_SIZE)  # 2 MB
        size = _dir_size_mb(f)
        assert size == pytest.approx(2.0, abs=APPROX_TOL)

    def test_directory(self, tmp_path):
        (tmp_path / "a.bin").write_bytes(b"x" * MB_SINGLE)
        (tmp_path / "b.bin").write_bytes(b"x" * MB_SINGLE)
        size = _dir_size_mb(tmp_path)
        assert size == pytest.approx(2.0, abs=APPROX_TOL)

    def test_empty_dir(self, tmp_path):
        assert _dir_size_mb(tmp_path) == 0.0

class TestQuantizeCLI:
    def test_dry_run_awq(self, tmp_path):
        model_path = tmp_path / "model"
        model_path.mkdir()
        out_dir = tmp_path / "quantized"

        with patch("sys.argv", [
            "quantize_model",
            "--model-path", str(model_path),
            "--method", "awq",
            "--bits", str(BITS_AWQ),
            "--dry-run",
            "--out-dir", str(out_dir),
        ]):
            ret = quantize_main()
        assert ret == 0

        plan = json.loads((out_dir / "quantize_plan.json").read_text())
        assert plan["method"] == "awq"
        assert plan["bits"] == BITS_AWQ

    def test_dry_run_gguf(self, tmp_path):
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
            ret = quantize_main()
        assert ret == 0

        plan = json.loads((out_dir / "quantize_plan.json").read_text())
        assert plan["method"] == "gguf"
        assert plan["gguf_quant"] == "q5_k_m"
        assert plan["bits"] is None

class TestSupportedMethods:
    def test_methods_exist(self):
        assert "awq" in SUPPORTED_METHODS
        assert "gptq" in SUPPORTED_METHODS
        assert "gguf" in SUPPORTED_METHODS

    def test_gguf_quants_exist(self):
        assert "q4_k_m" in GGUF_QUANTS
        assert "q8_0" in GGUF_QUANTS
        assert "f16" in GGUF_QUANTS

# ---------------------------------------------------------------------------
# Distillation tests
# ---------------------------------------------------------------------------

class TestDistillationStats:
    def test_defaults(self):
        stats = DistillationStats(
            teacher_model="Qwen/Qwen2.5-72B",
            student_model="Qwen/Qwen2.5-7B",
        )
        assert stats.kd_alpha == KD_ALPHA_DEFAULT
        assert stats.kd_beta == KD_BETA_DEFAULT
        assert stats.temperature == KD_TEMPERATURE_DEFAULT
        assert stats.epochs == 0
        assert stats.final_loss is None

    def test_to_dict(self):
        stats = DistillationStats(
            teacher_model="teacher",
            student_model="student",
            epochs=KD_EPOCHS,
            train_samples=KD_TRAIN_SAMPLES,
            final_loss=KD_FINAL_LOSS,
        )
        d = stats.to_dict()
        assert d["epochs"] == KD_EPOCHS
        assert d["train_samples"] == KD_TRAIN_SAMPLES
        assert d["final_loss"] == KD_FINAL_LOSS

class TestKDLoss:
    def test_returns_three_values(self):
        """Test that kd_loss returns (total, kd_loss, ce_loss) tuple."""
        batch, seq_len, vocab = BATCH_SMALL, SEQ_LEN, VOCAB_SMALL
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
        batch, seq_len, vocab = BATCH_SINGLE, SEQ_LEN_SHORT, VOCAB_TINY
        student_logits = torch.randn(batch, seq_len, vocab)
        teacher_logits = torch.randn(batch, seq_len, vocab)
        labels = torch.randint(0, vocab, (batch, seq_len))

        # With alpha=1, beta=0: total should equal kd loss.
        total_kd_only, kd, _ce = kd_loss(
            student_logits, teacher_logits, labels, alpha=1.0, beta=0.0
        )
        assert torch.allclose(total_kd_only, kd, atol=ATOL_LOSS)

        # With alpha=0, beta=1: total should equal ce loss.
        total_ce_only, _kd, ce = kd_loss(
            student_logits, teacher_logits, labels, alpha=0.0, beta=1.0
        )
        assert torch.allclose(total_ce_only, ce, atol=ATOL_LOSS)

class TestDistillCLI:
    def test_train_dry_run(self, tmp_path):
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
            ret = distill_main()
        assert ret == 0

        plan = json.loads((out_dir / "distill_plan.json").read_text())
        assert plan["student_model"] == "Qwen/Qwen2.5-7B"
        assert plan["kd_alpha"] == KD_ALPHA_DEFAULT
        assert plan["kd_beta"] == KD_BETA_DEFAULT

    def test_missing_command(self):
        with patch("sys.argv", ["distill_model"]), pytest.raises(SystemExit):
            distill_main()
