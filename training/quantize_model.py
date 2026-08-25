#!/usr/bin/env python3
"""Post-training quantization for inference deployment (PIX-4345 Appendix E Step 7).

Converts a fine-tuned model (base + merged LoRA adapter) to one of three
inference-optimized formats:

  - **AWQ**  — Activation-aware Weight Quantization (4-bit).  Best
    quality-to-size ratio; runs on vLLM.
  - **GPTQ** — Post-training quantization via second-order error compensation.
    Widely supported in vLLM / text-generation-inference.
  - **GGUF** — llama.cpp format (Q4_K_M, Q5_K_M, Q8_0).  Best for CPU / edge
    deployment via llama.cpp or Ollama.

After conversion, optionally benchmarks latency (tokens/sec) vs the
unquantized baseline.

Blueprint ref:
  - Appendix E  Step 7 (quantization for inference)
  - §2          (cost optimization: quantization post-training)
  - Appendix C  (AWQ / GPTQ / GGUF for deployment)

Usage (AWQ)::

    python -m ai.training.quantize_model \
        --model-path ./merged-model \
        --method awq \
        --bits 4 \
        --out-dir ./quantized/awq

Usage (GGUF, requires llama.cpp)::

    python -m ai.training.quantize_model \
        --model-path ./merged-model \
        --method gguf \
        --gguf-quant q4_k_m \
        --out-dir ./quantized/gguf

Usage (dry-run / plan only)::

    python -m ai.training.quantize_model \
        --model-path ./merged-model \
        --method awq \
        --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Supported quantization methods.
SUPPORTED_METHODS = ("awq", "gptq", "gguf")

# Supported GGUF quantization levels.
GGUF_QUANTS = (
    "q4_0",
    "q4_k_m",
    "q4_k_s",
    "q5_0",
    "q5_k_m",
    "q5_k_s",
    "q6_k",
    "q8_0",
    "f16",
    "f32",
)


@dataclass
class QuantizationStats:
    """Summary of a quantization run."""

    model_path: str
    method: str
    bits: int | None = None
    gguf_quant: str | None = None
    output_path: str = ""
    original_size_mb: float = 0.0
    quantized_size_mb: float = 0.0
    compression_ratio: float = 0.0
    latency_baseline_tps: float | None = None  # tokens/sec (unquantized)
    latency_quantized_tps: float | None = None  # tokens/sec (quantized)
    speedup: float | None = None
    quality_eval_score: float | None = None
    quality_loss: float | None = None
    quality_gate_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# AWQ quantization
# ---------------------------------------------------------------------------

def quantize_awq(
    model_path: str,
    out_dir: Path,
    bits: int = 4,
    group_size: int = 128,
) -> str:
    """Quantize a model to AWQ format.

    Uses the ``autoawq`` library.  The output is a directory of quantized
    weights in HuggingFace format, loadable by vLLM with ``--quantization awq``.

    Returns the output path.
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    logger.info("AWQ quantization: %s → %d-bit (group_size=%d)", model_path, bits, group_size)

    # AWQ config.
    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": bits,
        "version": "GEMM",
    }

    model = AutoAWQForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize.
    model.quantize(tokenizer, quant_config=quant_config)

    # Save.
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    logger.info("AWQ model saved to %s", out_dir)
    return str(out_dir)


# ---------------------------------------------------------------------------
# GPTQ quantization
# ---------------------------------------------------------------------------

def quantize_gptq(
    model_path: str,
    out_dir: Path,
    bits: int = 4,
    group_size: int = 128,
    calibration_samples: int = 128,
) -> str:
    """Quantize a model to GPTQ format.

    Uses the ``auto_gptq`` library.  The output is loadable by vLLM with
    ``--quantization gptq``.

    Returns the output path.
    """
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from datasets import load_dataset
    from transformers import AutoTokenizer

    logger.info("GPTQ quantization: %s → %d-bit (group_size=%d)", model_path, bits, group_size)

    quant_config = BaseQuantizeConfig(
        bits=bits,
        group_size=group_size,
        desc_act=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Calibration data: use a small slice of a general instruction dataset.
    calibration = load_dataset("c4", split="train", streaming=True)
    calib_samples: list[str] = []
    for i, sample in enumerate(calibration):
        if i >= calibration_samples:
            break
        calib_samples.append(sample["text"])

    model = AutoGPTQForCausalLM.from_pretrained(model_path, quant_config)
    model.quantize(calib_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out_dir), use_safetensors=True)
    tokenizer.save_pretrained(str(out_dir))

    logger.info("GPTQ model saved to %s", out_dir)
    return str(out_dir)


# ---------------------------------------------------------------------------
# GGUF quantization (llama.cpp)
# ---------------------------------------------------------------------------

def quantize_gguf(
    model_path: str,
    out_dir: Path,
    gguf_quant: str = "q4_k_m",
) -> str:
    """Convert a model to GGUF format and quantize via llama.cpp.

    Requires ``llama.cpp`` to be built locally (``LLAMA_CPP_DIR`` env var)
    or the ``llama-cpp-python`` package with build tools.

    Steps:
      1. Convert HF model → GGUF (unquantized f16).
      2. Quantize GGUF → target quant level.

    Returns the output path (the .gguf file).
    """
    import os
    import subprocess

    llama_dir = os.environ.get("LLAMA_CPP_DIR", "")
    if not llama_dir or not Path(llama_dir).exists():
        raise FileNotFoundError(
            "LLAMA_CPP_DIR not set or does not exist. "
            "Set it to your llama.cpp build directory, e.g. "
            "export LLAMA_CPP_DIR=/opt/llama.cpp"
        )

    llama_dir_path = Path(llama_dir)
    convert_script = llama_dir_path / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir_path / "build" / "bin" / "llama-quantize"

    if not convert_script.exists():
        raise FileNotFoundError(f"Conversion script not found: {convert_script}")
    if not quantize_bin.exists():
        raise FileNotFoundError(f"llama-quantize binary not found: {quantize_bin}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert HF → GGUF (f16).
    gguf_f16 = out_dir / "model_f16.gguf"
    logger.info("Converting HF model → GGUF (f16): %s", gguf_f16)
    subprocess.run(
        [
            sys.executable,
            str(convert_script),
            model_path,
            str(gguf_f16),
            "--outtype", "f16",
        ],
        check=True,
    )

    # Step 2: Quantize to target level.
    gguf_out = out_dir / f"model_{gguf_quant}.gguf"
    logger.info("Quantizing GGUF → %s: %s", gguf_quant, gguf_out)
    subprocess.run(
        [
            str(quantize_bin),
            str(gguf_f16),
            str(gguf_out),
            gguf_quant,
        ],
        check=True,
    )

    # Clean up the intermediate f16 file.
    gguf_f16.unlink(missing_ok=True)

    logger.info("GGUF model saved to %s", gguf_out)
    return str(gguf_out)


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(
    model_path: str,
    prompt: str = "Explain the concept of cognitive behavioral therapy in 3 sentences.",
    max_new_tokens: int = 128,
    warmup: int = 1,
    rounds: int = 3,
) -> float:
    """Benchmark tokens/sec for a model via vLLM or direct HF generate.

    Returns the average tokens/sec across ``rounds`` runs.
    """
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warmup.
    for _ in range(warmup):
        with torch.inference_mode():
            model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    # Timed runs.
    tps_list: list[float] = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        elapsed = time.perf_counter() - t0
        generated_tokens = out.shape[-1] - inputs["input_ids"].shape[-1]
        tps = generated_tokens / elapsed if elapsed > 0 else 0.0
        tps_list.append(tps)

    avg_tps = sum(tps_list) / len(tps_list)
    logger.info("Latency benchmark: %.1f tokens/sec (avg of %d rounds)", avg_tps, rounds)
    return round(avg_tps, 2)


# ---------------------------------------------------------------------------
# File size helpers
# ---------------------------------------------------------------------------

def _dir_size_mb(path: Path) -> float:
    """Total size of all files in a directory (or single file) in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return round(total / (1024 * 1024), 2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-training quantization for inference (PIX-4345 Appendix E Step 7)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the merged model (base + LoRA adapter merged)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=SUPPORTED_METHODS,
        required=True,
        help="Quantization method: awq, gptq, or gguf",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Quantization bits (AWQ/GPTQ). Default: 4",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="Group size for AWQ/GPTQ. Default: 128",
    )
    parser.add_argument(
        "--gguf-quant",
        type=str,
        default="q4_k_m",
        choices=GGUF_QUANTS,
        help="GGUF quantization level. Default: q4_k_m",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ai/training/quantized",
        help="Output directory for quantized model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark (tokens/sec) after quantization",
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default=None,
        help="Path to unquantized baseline model for latency comparison",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip actual quantization; emit a plan with estimated sizes",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    out_dir = Path(args.out_dir)

    if args.dry_run:
        stats = QuantizationStats(
            model_path=args.model_path,
            method=args.method,
            bits=args.bits if args.method != "gguf" else None,
            gguf_quant=args.gguf_quant if args.method == "gguf" else None,
            output_path=str(out_dir),
        )
        report_path = out_dir / "quantize_plan.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"[quantize] dry-run plan written to {report_path}")
        print(f"[quantize] method={args.method} bits={args.bits} gguf_quant={args.gguf_quant}")
        return 0

    # Original model size.
    model_path = Path(args.model_path)
    original_size = _dir_size_mb(model_path)
    logger.info("Original model size: %.1f MB", original_size)

    # Quantize.
    if args.method == "awq":
        output = quantize_awq(args.model_path, out_dir, bits=args.bits, group_size=args.group_size)
    elif args.method == "gptq":
        output = quantize_gptq(args.model_path, out_dir, bits=args.bits, group_size=args.group_size)
    elif args.method == "gguf":
        output = quantize_gguf(args.model_path, out_dir, gguf_quant=args.gguf_quant)
    else:
        print(f"[quantize] ERROR: unsupported method '{args.method}'")
        return 1

    output_path = Path(output)
    quantized_size = _dir_size_mb(output_path)
    compression_ratio = original_size / quantized_size if quantized_size > 0 else 0.0

    stats = QuantizationStats(
        model_path=args.model_path,
        method=args.method,
        bits=args.bits if args.method != "gguf" else None,
        gguf_quant=args.gguf_quant if args.method == "gguf" else None,
        output_path=str(output_path),
        original_size_mb=original_size,
        quantized_size_mb=quantized_size,
        compression_ratio=round(compression_ratio, 2),
    )

    # Latency benchmark.
    if args.benchmark:
        if args.baseline_model:
            logger.info("Benchmarking baseline model: %s", args.baseline_model)
            stats.latency_baseline_tps = benchmark_latency(args.baseline_model)

        logger.info("Benchmarking quantized model: %s", str(output_path))
        stats.latency_quantized_tps = benchmark_latency(str(output_path))

        if stats.latency_baseline_tps and stats.latency_quantized_tps:
            stats.speedup = round(stats.latency_quantized_tps / stats.latency_baseline_tps, 2)

    # Write report.
    report_path = out_dir / "quantize_report.json"
    report_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    print(f"[quantize] report written to {report_path}")
    print(f"[quantize] {args.method}: {original_size:.1f} MB → {quantized_size:.1f} MB ({compression_ratio:.1f}x compression)")

    if stats.speedup is not None:
        print(f"[quantize] latency: {stats.latency_quantized_tps:.1f} tps (baseline: {stats.latency_baseline_tps:.1f} tps, {stats.speedup:.2f}x speedup)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
