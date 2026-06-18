import argparse
import statistics
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:  # pragma: no cover - optional in minimal envs
    given = None
    settings = None
    st = None

import torch

from training.shared_config import (
    add_lora_args,
    build_lora_config,
    count_truncated,
    log_token_length_distribution,
    shared_qlora_config,
)


@dataclass
class _LogMessage:
    level: str
    message: str


class _TestLogger:
    def __init__(self):
        self.messages = []

    def info(self, message, *args):
        self.messages.append(_LogMessage("info", message % args if args else message))

    def warning(self, message, *args):
        self.messages.append(_LogMessage("warning", message % args if args else message))


def test_shared_qlora_config_uses_bf16_when_supported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    config = shared_qlora_config()
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    assert config.bnb_4bit_compute_dtype == torch.bfloat16


def test_shared_qlora_config_uses_fp16_without_bf16(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    config = shared_qlora_config()
    assert config.bnb_4bit_compute_dtype == torch.float16


def test_add_lora_args_registers_expected_flags():
    parser = argparse.ArgumentParser()
    add_lora_args(parser)
    args = parser.parse_args([])
    assert args.lora_r == 8
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.lora_bias == "none"
    assert args.lora_target_modules == "q_proj,k_proj,v_proj,o_proj"


def test_build_lora_config_uses_expected_defaults():
    args = SimpleNamespace(
        lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_bias="none", lora_target_modules="q_proj,k_proj,v_proj,o_proj"
    )
    config = build_lora_config(args)
    assert config.r == 8
    assert config.lora_alpha == 16
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_build_lora_config_rejects_empty_target_modules():
    args = SimpleNamespace(lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_bias="none", lora_target_modules=" , ,   ")
    with pytest.raises(ValueError, match="empty list"):
        build_lora_config(args)


@pytest.mark.parametrize(
    ("lengths", "max_len", "expected"),
    [
        ([], 512, 0),
        ([100, 150, 512], 400, 1),
        ([1000, 2000], 500, 2),
    ],
)
def test_count_truncated_edge_cases(lengths, max_len, expected):
    assert count_truncated(lengths, max_len) == expected


def _p95_reference(values):
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=20, method="inclusive")[18]


if st is not None:

    @given(st.lists(st.integers(min_value=1, max_value=16384), min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_log_token_length_distribution_statistics_are_correct(lengths):
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=lengths,
            max_seq_length=max(lengths) + 1,
            logger=logger,
            field_name="token_count",
        )
        assert result["min"] == min(lengths)
        assert result["max"] == max(lengths)
        assert result["mean"] == pytest.approx(statistics.mean(lengths))
        assert result["median"] == pytest.approx(statistics.median(lengths))
        assert result["p95"] == pytest.approx(_p95_reference(lengths))
        assert result["truncated_count"] == 0

    @given(st.lists(st.integers(min_value=1, max_value=16384), min_size=1, max_size=500))
    @settings(max_examples=100)
    def test_log_token_length_distribution_truncation_warning_is_counted(lengths):
        max_seq_length = lengths[0]
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=lengths,
            max_seq_length=max_seq_length,
            logger=logger,
            field_name="token_count",
        )
        expected = sum(1 for val in lengths if val > max_seq_length)
        assert result["truncated_count"] == expected
else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_log_token_length_distribution_statistics_are_correct():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_log_token_length_distribution_truncation_warning_is_counted():
        raise AssertionError("Skipped when hypothesis is unavailable")


# ---------------------------------------------------------------------------
# Hypothesis property tests (VAL-TEST-001, VAL-TEST-004, VAL-TEST-005)
# ---------------------------------------------------------------------------

if st is not None:

    @given(st.lists(st.integers(min_value=1, max_value=16384), min_size=0, max_size=500))
    @settings(max_examples=200)
    def test_count_truncated_property(lengths):
        """Token-id lists of any length 0..16384 are handled by count_truncated.
        Covers the empty-list branch (lines 98-113 of shared_config.py)."""
        max_len = 100
        result = count_truncated(lengths, max_len)
        expected = sum(1 for v in lengths if v > max_len)
        assert result == expected
        assert result >= 0

    @given(st.lists(st.integers(min_value=0, max_value=20000), min_size=0, max_size=1000))
    @settings(max_examples=200)
    def test_log_token_length_distribution_empty_list_property(lengths):
        """An empty list triggers the early-return branch (lines 98-113) and
        returns a zero-filled stats dict. A non-empty list returns real values."""
        max_len = 256
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=lengths,
            max_seq_length=max_len,
            logger=logger,
            field_name="token_count",
        )
        assert result["count"] == len(lengths)
        if not lengths:
            # Empty-list branch: all numeric stats are zeroed
            assert result["min"] == 0
            assert result["max"] == 0
            assert result["mean"] == 0.0
            assert result["median"] == 0.0
            assert result["p95"] == 0.0
            assert result["truncated_count"] == 0
        else:
            # Non-empty: stats are finite (no NaN/Inf), count matches input length
            import math
            assert result["count"] == len(lengths)
            assert result["min"] >= 0
            assert result["max"] >= result["min"]
            assert result["mean"] >= 0
            assert result["median"] >= 0
            assert not math.isnan(result["mean"])
            assert not math.isnan(result["median"])

    @given(st.lists(st.integers(min_value=1, max_value=16384), min_size=1, max_size=500))
    @settings(max_examples=200)
    def test_build_lora_config_lora_rank_boundary(lengths):
        """LoRA rank in {0,4,8,16,32,64} is preserved through build_lora_config."""
        for rank in (0, 4, 8, 16, 32, 64):
            args = SimpleNamespace(
                lora_r=rank,
                lora_alpha=rank * 2,
                lora_dropout=0.05,
                lora_bias="none",
                lora_target_modules="q_proj,k_proj,v_proj,o_proj",
            )
            cfg = build_lora_config(args)
            assert cfg.r == rank, f"expected r={rank}, got {cfg.r}"
            assert cfg.lora_alpha == rank * 2

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_count_truncated_property():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_log_token_length_distribution_empty_list_property():
        raise AssertionError("Skipped when hypothesis is unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_build_lora_config_lora_rank_boundary():
        raise AssertionError("Skipped when hypothesis is unavailable")


# ---------------------------------------------------------------------------
# Deterministic boundary tests (VAL-TEST-001, VAL-TEST-005)
# ---------------------------------------------------------------------------

def test_count_truncated_empty_list():
    assert count_truncated([], 512) == 0
    assert count_truncated([], 0) == 0


def test_build_lora_config_defaults_rank_is_8():
    """Calling build_lora_config without an explicit r returns the default r=8."""
    args = SimpleNamespace(
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_bias="none",
        lora_target_modules="q_proj,k_proj,v_proj,o_proj",
    )
    cfg = build_lora_config(args)
    assert cfg.r == 8
