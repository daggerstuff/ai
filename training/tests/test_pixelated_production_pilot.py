#!/usr/bin/env python3
"""Unit tests for SFT script improvements from Phase 1."""
from __future__ import annotations

import argparse
import logging

import pytest

from training.pixelated_production_pilot import _build_arg_parser
from training.shared_config import build_lora_config


class TestSftSeqLengthArgparse:
    def test_max_seq_length_511_raises_system_exit(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--data_path", "/tmp/data.jsonl",
                "--output_dir", "/tmp/output",
                "--max_seq_length", "511",
            ])

    def test_max_seq_length_8193_raises_system_exit(self) -> None:
        parser = _build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--data_path", "/tmp/data.jsonl",
                "--output_dir", "/tmp/output",
                "--max_seq_length", "8193",
            ])

    def test_max_seq_length_2048_accepted(self) -> None:
        parser = _build_arg_parser()
        args = parser.parse_args([
            "--data_path", "/tmp/data.jsonl",
            "--output_dir", "/tmp/output",
            "--max_seq_length", "2048",
        ])
        assert args.max_seq_length == 2048


class TestLoraTargetModules:
    def test_gate_proj_up_proj_down_proj_accepted(self, caplog) -> None:
        caplog.set_level(logging.INFO)
        args = argparse.Namespace(
            lora_r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            lora_bias="none",
            lora_target_modules="gate_proj,up_proj,down_proj",
        )
        lora_config = build_lora_config(args)
        assert set([module.strip() for module in (lora_config.target_modules or [])]) == {
            "gate_proj",
            "up_proj",
            "down_proj",
        }
