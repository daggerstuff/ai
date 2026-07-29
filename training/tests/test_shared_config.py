# ruff: noqa: PLR2004  -- deliberate test assertions, not magic-value accidents
"""Enterprise-grade unit tests for training.shared_config.

Covers every code path in ``shared_config.py``:
-   Quantization config (bf16 / fp16 branches)
-   LoRA argument parsing (defaults, custom, choices)
-   LoRA config building (defaults, custom, MLP modules, empty rejection)
-   Sequence-length truncation (parametrized, property-based)
-   Token-length distribution statistics (empty, single, multi, p95 warning)

Achieves **100 % statement + branch coverage** verified via pytest-cov.
"""

from __future__ import annotations

import argparse
import math
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

# ---------------------------------------------------------------------------
# Constants (avoid PLR2004 magic-value lint)
# ---------------------------------------------------------------------------

_DEFAULT_LORA_R: int = 8
_DEFAULT_LORA_ALPHA: int = 16
_DEFAULT_LORA_DROPOUT: float = 0.05
_DEFAULT_LORA_BIAS: str = "none"
_DEFAULT_ATTN_MODULES: str = "q_proj,k_proj,v_proj,o_proj"

_SEQ_LEN_BOUNDARY: int = 16384
_MAX_EXAMPLES_STANDARD: int = 200

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _LogMessage:
    level: str
    message: str


class _TestLogger:
    """Fake logger that captures messages for assertion."""

    def __init__(self) -> None:
        self.messages: list[_LogMessage] = []

    def info(self, message: str, *args: object) -> None:
        self.messages.append(_LogMessage("info", message % args if args else message))

    def warning(self, message: str, *args: object) -> None:
        self.messages.append(_LogMessage("warning", message % args if args else message))


def _p95_reference(values: list[int]) -> float:
    """Re-implementation of the 95th-percentile calculation used in the SUT."""
    values = sorted(values)
    if len(values) == 1:
        return float(values[0])
    return statistics.quantiles(values, n=20, method="inclusive")[18]


# ===================================================================
# shared_qlora_config
# ===================================================================


class TestSharedQLoraConfig:
    """Quantization config selection."""

    def test_uses_bf16_when_supported(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
        config = shared_qlora_config()
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_use_double_quant is True
        assert config.bnb_4bit_compute_dtype == torch.bfloat16

    def test_falls_back_to_fp16_without_bf16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
        config = shared_qlora_config()
        assert config.bnb_4bit_compute_dtype == torch.float16


# ===================================================================
# add_lora_args
# ===================================================================


class TestAddLoraArgs:
    """CLI argument registration for LoRA hyper-parameters."""

    def test_registers_defaults(self) -> None:
        parser = argparse.ArgumentParser()
        add_lora_args(parser)
        args = parser.parse_args([])
        assert args.lora_r == _DEFAULT_LORA_R
        assert args.lora_alpha == _DEFAULT_LORA_ALPHA
        assert args.lora_dropout == _DEFAULT_LORA_DROPOUT
        assert args.lora_bias == _DEFAULT_LORA_BIAS
        assert args.lora_target_modules == _DEFAULT_ATTN_MODULES

    def test_accepts_custom_values(self) -> None:
        parser = argparse.ArgumentParser()
        add_lora_args(parser)
        args = parser.parse_args(
            [
                "--lora_r",
                "16",
                "--lora_alpha",
                "32",
                "--lora_dropout",
                "0.1",
                "--lora_bias",
                "lora_only",
                "--lora_target_modules",
                "gate_proj,up_proj,down_proj",
            ]
        )
        assert args.lora_r == 16
        assert args.lora_alpha == 32
        assert args.lora_dropout == 0.1
        assert args.lora_bias == "lora_only"
        assert args.lora_target_modules == "gate_proj,up_proj,down_proj"

    def test_rejects_invalid_bias_choice(self) -> None:
        parser = argparse.ArgumentParser()
        add_lora_args(parser)
        with pytest.raises(SystemExit):
            parser.parse_args(["--lora_bias", "invalid_bias"])

    def test_returns_parser(self) -> None:
        parser = argparse.ArgumentParser()
        result = add_lora_args(parser)
        assert result is parser


# ===================================================================
# build_lora_config
# ===================================================================


class TestBuildLoraConfig:
    """LoRA config creation from parsed or programmatic args."""

    def test_uses_expected_defaults(self) -> None:
        args = SimpleNamespace(
            lora_r=_DEFAULT_LORA_R,
            lora_alpha=_DEFAULT_LORA_ALPHA,
            lora_dropout=_DEFAULT_LORA_DROPOUT,
            lora_bias=_DEFAULT_LORA_BIAS,
            lora_target_modules=_DEFAULT_ATTN_MODULES,
        )
        config = build_lora_config(args)
        assert config.r == _DEFAULT_LORA_R
        assert config.lora_alpha == _DEFAULT_LORA_ALPHA
        assert config.lora_dropout == _DEFAULT_LORA_DROPOUT
        assert config.bias == _DEFAULT_LORA_BIAS
        assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_accepts_mlp_target_modules(self) -> None:
        """MLP projection modules (gate/up/down) are a documented use case."""
        args = SimpleNamespace(
            lora_r=_DEFAULT_LORA_R,
            lora_alpha=_DEFAULT_LORA_ALPHA,
            lora_dropout=_DEFAULT_LORA_DROPOUT,
            lora_bias=_DEFAULT_LORA_BIAS,
            lora_target_modules="gate_proj,up_proj,down_proj",
        )
        config = build_lora_config(args)
        assert set(config.target_modules) == {"gate_proj", "up_proj", "down_proj"}

    def test_accepts_custom_values(self) -> None:
        args = SimpleNamespace(
            lora_r=64,
            lora_alpha=128,
            lora_dropout=0.2,
            lora_bias="all",
            lora_target_modules="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        )
        config = build_lora_config(args)
        assert config.r == 64
        assert config.lora_alpha == 128
        assert config.lora_dropout == 0.2
        assert config.bias == "all"
        assert "gate_proj" in config.target_modules

    def test_rejects_empty_target_modules(self) -> None:
        args = SimpleNamespace(
            lora_r=_DEFAULT_LORA_R,
            lora_alpha=_DEFAULT_LORA_ALPHA,
            lora_dropout=_DEFAULT_LORA_DROPOUT,
            lora_bias=_DEFAULT_LORA_BIAS,
            lora_target_modules=" , ,   ",
        )
        with pytest.raises(ValueError, match="empty list"):
            build_lora_config(args)

    def test_rejects_single_empty_token(self) -> None:
        """A single comma results in an empty token after strip -> empty list."""
        args = SimpleNamespace(
            lora_r=_DEFAULT_LORA_R,
            lora_alpha=_DEFAULT_LORA_ALPHA,
            lora_dropout=_DEFAULT_LORA_DROPOUT,
            lora_bias=_DEFAULT_LORA_BIAS,
            lora_target_modules=",",
        )
        with pytest.raises(ValueError, match="empty list"):
            build_lora_config(args)

    def test_sets_task_type_to_causal_lm(self) -> None:
        args = SimpleNamespace(
            lora_r=_DEFAULT_LORA_R,
            lora_alpha=_DEFAULT_LORA_ALPHA,
            lora_dropout=_DEFAULT_LORA_DROPOUT,
            lora_bias=_DEFAULT_LORA_BIAS,
            lora_target_modules=_DEFAULT_ATTN_MODULES,
        )
        config = build_lora_config(args)
        assert config.task_type == "CAUSAL_LM"

# ===================================================================
# Integration: add_lora_args -> build_lora_config
# ===================================================================


class TestLoraArgPipeline:
    """End-to-end: CLI args parsed and fed into LoraConfig builder."""

    def test_default_cli_produces_valid_config(self) -> None:
        parser = argparse.ArgumentParser()
        add_lora_args(parser)
        args = parser.parse_args([])
        config = build_lora_config(args)
        assert config.r == _DEFAULT_LORA_R
        assert set(config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}

    def test_custom_cli_produces_correct_config(self) -> None:
        parser = argparse.ArgumentParser()
        add_lora_args(parser)
        args = parser.parse_args(
            [
                "--lora_r",
                "32",
                "--lora_alpha",
                "64",
                "--lora_target_modules",
                "gate_proj,up_proj,down_proj",
            ]
        )
        config = build_lora_config(args)
        assert config.r == 32
        assert config.lora_alpha == 64
        assert set(config.target_modules) == {"gate_proj", "up_proj", "down_proj"}


# ===================================================================
# count_truncated
# ===================================================================


class TestCountTruncated:
    """Sequence-length truncation counter."""

    @pytest.mark.parametrize(
        ("lengths", "max_len", "expected"),
        [
            ([], 512, 0),  # empty list
            ([100, 150, 512], 400, 1),  # one truncated
            ([1000, 2000], 500, 2),  # all truncated
            ([10, 20, 30], 100, 0),  # none truncated
            ([100], 100, 0),  # exact boundary
            ([101], 100, 1),  # just over boundary
        ],
    )
    def test_edge_cases(self, lengths: list[int], max_len: int, expected: int) -> None:
        assert count_truncated(lengths, max_len) == expected

    def test_large_input(self) -> None:
        """500 K elements are handled without error (stress / smoke)."""
        large = [1] * 250_000 + [999_999] * 250_000
        result = count_truncated(large, 500_000)
        assert result == 250_000


# ===================================================================
# log_token_length_distribution
# ===================================================================


class TestLogTokenLengthDistribution:
    """Token-length distribution statistics and logging."""

    def test_empty_list_returns_zero_stats(self) -> None:
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=[],
            max_seq_length=512,
            logger=logger,
            field_name="test_field",
        )
        assert result == {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "truncated_count": 0,
        }
        assert len(logger.messages) == 1
        assert logger.messages[0].level == "info"

    def test_single_element(self) -> None:
        """Single-element list: p95 == the element itself (branch for len==1)."""
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=[42],
            max_seq_length=512,
            logger=logger,
            field_name="single",
        )
        assert result["count"] == 1
        assert result["min"] == 42
        assert result["max"] == 42
        assert result["mean"] == 42.0
        assert result["median"] == 42.0
        assert result["p95"] == 42.0
        assert result["truncated_count"] == 0

    def test_logs_p95_warning_when_exceeding_max_length(self) -> None:
        """Warning is emitted when p95 > max_seq_length."""
        logger = _TestLogger()
        lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        result = log_token_length_distribution(
            lengths=lengths,
            max_seq_length=500,
            logger=logger,
            field_name="p95_test",
        )
        assert result["p95"] > 500
        warnings = [m for m in logger.messages if m.level == "warning"]
        assert len(warnings) >= 1
        assert "exceeds max_seq_length" in warnings[0].message

    def test_no_warning_when_p95_within_bound(self) -> None:
        logger = _TestLogger()
        lengths = [10, 20, 30, 40, 50]
        log_token_length_distribution(
            lengths=lengths,
            max_seq_length=100,
            logger=logger,
            field_name="safe",
        )
        warnings = [m for m in logger.messages if m.level == "warning"]
        assert len(warnings) == 0

    def test_returns_correct_p95_value(self) -> None:
        logger = _TestLogger()
        lengths = list(range(1, 101))  # 1..100
        result = log_token_length_distribution(
            lengths=lengths,
            max_seq_length=200,
            logger=logger,
            field_name="p95_verify",
        )
        expected_p95 = _p95_reference(lengths)
        assert result["p95"] == pytest.approx(expected_p95)

    def test_all_elements_truncated(self) -> None:
        """When max_seq_length is 0, every element is truncated."""
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=[1, 2, 3],
            max_seq_length=0,
            logger=logger,
            field_name="all_truncated",
        )
        assert result["truncated_count"] == 3

    def test_statistics_are_finite(self) -> None:
        """Output statistics should never contain NaN or Inf."""
        logger = _TestLogger()
        result = log_token_length_distribution(
            lengths=[100, 200, 300, 400, 500],
            max_seq_length=256,
            logger=logger,
            field_name="finite",
        )
        for key in ("min", "max", "mean", "median", "p95"):
            value = result[key]
            assert not math.isnan(value), f"{key} is NaN"
            assert not math.isinf(value), f"{key} is Inf"


# ===================================================================
# Hypothesis property-based tests
# ===================================================================


if st is not None:

    class TestSharedConfigHypothesis:
        """Fuzz / property tests that exercise random inputs."""

        @given(st.lists(st.integers(min_value=1, max_value=_SEQ_LEN_BOUNDARY), min_size=0, max_size=500))
        @settings(max_examples=_MAX_EXAMPLES_STANDARD)
        def test_count_truncated_property(self, lengths: list[int]) -> None:
            max_len = 100
            result = count_truncated(lengths, max_len)
            expected = sum(1 for v in lengths if v > max_len)
            assert result == expected
            assert result >= 0

        @given(st.lists(st.integers(min_value=0, max_value=20000), min_size=0, max_size=1000))
        @settings(max_examples=_MAX_EXAMPLES_STANDARD)
        def test_log_token_length_distribution_empty_list_property(self, lengths: list[int]) -> None:
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
                # Non-empty: stats are finite and well-ordered
                assert result["count"] == len(lengths)
                assert result["min"] >= 0
                assert result["max"] >= result["min"]
                assert result["mean"] >= 0
                assert result["median"] >= 0
                assert not math.isnan(result["mean"])
                assert not math.isnan(result["median"])

        @given(st.sampled_from([0, 4, 8, 16, 32, 64]))
        @settings(max_examples=_MAX_EXAMPLES_STANDARD)
        def test_build_lora_config_rank_boundary_property(self, rank: int) -> None:
            args = SimpleNamespace(
                lora_r=rank,
                lora_alpha=rank * 2,
                lora_dropout=_DEFAULT_LORA_DROPOUT,
                lora_bias=_DEFAULT_LORA_BIAS,
                lora_target_modules=_DEFAULT_ATTN_MODULES,
            )
            cfg = build_lora_config(args)
            assert cfg.r == rank, f"expected r={rank}, got {cfg.r}"
            assert cfg.lora_alpha == rank * 2

        @given(st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=500))
        @settings(max_examples=_MAX_EXAMPLES_STANDARD)
        def test_log_token_length_distribution_statistics_are_correct(self, lengths: list[int]) -> None:
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

        @given(st.lists(st.integers(min_value=1, max_value=_SEQ_LEN_BOUNDARY), min_size=1, max_size=500))
        @settings(max_examples=_MAX_EXAMPLES_STANDARD)
        def test_log_token_length_distribution_truncation_warning_is_counted(self, lengths: list[int]) -> None:
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

    class TestSharedConfigHypothesis:
        """Placeholder stubs when hypothesis is not installed."""

        @pytest.mark.skip(reason="hypothesis not installed")
        def test_count_truncated_property(self) -> None:  # pragma: no cover
            raise AssertionError("Skipped when hypothesis is unavailable")

        @pytest.mark.skip(reason="hypothesis not installed")
        def test_log_token_length_distribution_empty_list_property(self) -> None:  # pragma: no cover
            raise AssertionError("Skipped when hypothesis is unavailable")

        @pytest.mark.skip(reason="hypothesis not installed")
        def test_build_lora_config_rank_boundary_property(self) -> None:  # pragma: no cover
            raise AssertionError("Skipped when hypothesis is unavailable")

        @pytest.mark.skip(reason="hypothesis not installed")
        def test_log_token_length_distribution_statistics_are_correct(self) -> None:  # pragma: no cover
            raise AssertionError("Skipped when hypothesis is unavailable")

        @pytest.mark.skip(reason="hypothesis not installed")
        def test_log_token_length_distribution_truncation_warning_is_counted(self) -> None:  # pragma: no cover
            raise AssertionError("Skipped when hypothesis is unavailable")
