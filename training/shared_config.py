"""Shared configuration helpers for training scripts."""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig


def shared_qlora_config() -> BitsAndBytesConfig:
    """Return a default 4-bit quantization config for training."""
    import torch
    from transformers import BitsAndBytesConfig

    # Using dynamic access to satisfy broken torch stubs without suppressions
    dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    compute_dtype = getattr(torch, dtype_name)

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def add_lora_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register common LoRA CLI arguments on ``parser``."""
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="LoRA bias training mode.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated list of target modules for LoRA. "
        "Attention: q_proj,k_proj,v_proj,o_proj. MLP: gate_proj,up_proj,down_proj.",
    )
    return parser


def build_lora_config(args: argparse.Namespace) -> LoraConfig:
    """Build a PEFT ``LoraConfig`` from parsed CLI args."""
    from peft import LoraConfig

    target_modules = [m.strip() for m in getattr(args, "lora_target_modules", "").split(",") if m.strip()]
    if not target_modules:
        raise ValueError("--lora_target_modules produced an empty list")

    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def count_truncated(lengths: Iterable[int], max_seq_length: int) -> int:
    """Count how many sequence lengths exceed ``max_seq_length``."""
    return sum(1 for length in lengths if length > max_seq_length)


def log_token_length_distribution(
    lengths: Iterable[int],
    max_seq_length: int,
    logger,
    field_name: str,
) -> dict[str, float]:
    """Compute and log token-length distribution stats.

    Args:
        lengths: Sequence lengths to analyze.
        max_seq_length: Configured training max sequence length.
        logger: Logger with ``info`` and ``warning`` methods.
        field_name: Human label used in logs.

    Returns:
        Dictionary with min/max/mean/median/p95/truncated_count.
    """
    lengths = list(lengths)
    truncated_count = count_truncated(lengths, max_seq_length)

    if not lengths:
        stats = {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "truncated_count": truncated_count,
        }
        logger.info(
            "%s: no samples to analyze; seq_len=%s (truncated=%s)",
            field_name,
            max_seq_length,
            truncated_count,
        )
        return stats

    sorted_lengths = sorted(lengths)
    min_value = sorted_lengths[0]
    max_value = sorted_lengths[-1]
    mean_value = statistics.mean(sorted_lengths)
    median_value = statistics.median(sorted_lengths)
    if len(sorted_lengths) > 1:
        p95_value = float(statistics.quantiles(sorted_lengths, n=20, method="inclusive")[18])
    else:
        p95_value = float(sorted_lengths[0])

    if p95_value > max_seq_length:
        logger.warning(
            "%s: p95=%s exceeds max_seq_length=%s. Likely truncation.",
            field_name,
            p95_value,
            max_seq_length,
        )

    logger.info(
        "%s: count=%s min=%s max=%s mean=%.2f median=%.2f p95=%.2f truncated=%s",
        field_name,
        len(sorted_lengths),
        min_value,
        max_value,
        mean_value,
        median_value,
        p95_value,
        truncated_count,
    )

    return {
        "count": len(sorted_lengths),
        "min": min_value,
        "max": max_value,
        "mean": mean_value,
        "median": median_value,
        "p95": p95_value,
        "truncated_count": truncated_count,
    }
