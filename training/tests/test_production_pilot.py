"""Tests for the production SFT script CLI argument validation.

Tests that require the full pilot module (with trl/peft/transformers) use
importorskip. Tests that only need shared_config run without ML deps.
"""

from __future__ import annotations

import argparse

import pytest

from training.shared_config import add_lora_args


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        choices=range(512, 8193),
        help="Max token sequence length.",
    )
    add_lora_args(parser)
    return parser


# ---------------------------------------------------------------------------
# --max_seq_length validation
# ---------------------------------------------------------------------------

def test_max_seq_length_valid_2048():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json", "--max_seq_length", "2048"])
    assert args.max_seq_length == 2048


def test_max_seq_length_default_1024():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json"])
    assert args.max_seq_length == 1024


def test_max_seq_length_below_range():
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--data_path", "/tmp/test.json", "--max_seq_length", "511"])


def test_max_seq_length_above_range():
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--data_path", "/tmp/test.json", "--max_seq_length", "8193"])


def test_max_seq_length_boundary_512():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json", "--max_seq_length", "512"])
    assert args.max_seq_length == 512


def test_max_seq_length_boundary_8192():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json", "--max_seq_length", "8192"])
    assert args.max_seq_length == 8192


# ---------------------------------------------------------------------------
# --lora_target_modules (MLP modules accepted)
# ---------------------------------------------------------------------------

def test_lora_target_modules_mlp():
    args = _make_parser().parse_args([
        "--data_path", "/tmp/test.json",
        "--lora_target_modules", "gate_proj,up_proj,down_proj",
    ])
    assert args.lora_target_modules == "gate_proj,up_proj,down_proj"


def test_lora_target_modules_default():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json"])
    assert args.lora_target_modules == "q_proj,k_proj,v_proj,o_proj"


def test_lora_args_registered():
    args = _make_parser().parse_args(["--data_path", "/tmp/test.json"])
    assert args.lora_r == 8
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.05
    assert args.lora_bias == "none"


# ---------------------------------------------------------------------------
# Full pilot module tests (require trl/peft/transformers)
# ---------------------------------------------------------------------------

def test_safety_checker_disabled():
    """SAFETY CHECKERS DISABLED per user request — SAFETY_CHECKER is None."""
    pilot = pytest.importorskip("training.pixelated_production_pilot")
    assert pilot.SAFETY_CHECKER is None


def test_shared_config_imports_available():
    pilot = pytest.importorskip("training.pixelated_production_pilot")
    assert callable(pilot.shared_qlora_config)
    assert callable(pilot.add_lora_args)
    assert callable(pilot.build_lora_config)
    assert callable(pilot.log_token_length_distribution)
