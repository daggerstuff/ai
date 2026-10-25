"""Validate private campaign exports on CPU without loading base-model weights."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from ai.training.backup_dual_model_sft_checkpoints import sha256_file

BASES = {
    "gemma4": {"google/gemma-4-26B-A4B-it"},
    "nemotron35": {
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
    },
}
MATRIX_DIMENSIONS = 2


def verified_manifest(directory: Path) -> dict[str, Any]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must be an object")
    names = set()
    for entry in manifest["local_files"]:
        path = directory / entry["name"]
        if not path.resolve().is_relative_to(directory.resolve()) or entry["name"] in names:
            raise ValueError("manifest file path escapes the adapter or repeats")
        names.add(entry["name"])
        if path.stat().st_size != entry["size_bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("adapter file differs from preserved manifest")
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(names):
        raise ValueError("manifest does not bind both adapter configuration and weights")
    return manifest


def tensor_summary(path: Path, expected_rank: int, packed_experts: int = 1) -> dict[str, Any]:
    dtypes: Counter[str] = Counter()
    elements = 0
    nonzero_b = 0
    shapes = {}
    pair_ranks: Counter[int] = Counter()
    with safe_open(path, framework="pt", device="cpu") as tensors:
        names = tensors.keys()
        for name in names:
            tensor = tensors.get_tensor(name)
            if tensor.ndim != MATRIX_DIMENSIONS or not tensor.numel() or not tensor.is_floating_point():
                raise ValueError("unsupported LoRA tensor shape or dtype")
            if not torch.isfinite(tensor).all().item():
                raise ValueError("adapter contains nonfinite tensor values")
            if not name.endswith((".lora_A.weight", ".lora_B.weight")):
                raise ValueError("adapter contains an unexpected tensor name")
            dtypes[str(tensor.dtype)] += 1
            elements += tensor.numel()
            shapes[name] = list(tensor.shape)
            if name.endswith(".lora_B.weight"):
                nonzero_b += torch.count_nonzero(tensor).item()
    if not shapes:
        raise ValueError("adapter has no tensors")
    for name, shape in shapes.items():
        if name.endswith(".lora_A.weight"):
            partner = name.removesuffix(".lora_A.weight") + ".lora_B.weight"
            if partner not in shapes or shape[0] != shapes[partner][1]:
                raise ValueError("LoRA A/B tensors are missing or have incompatible ranks")
            # ART fused expert exports pack num_experts * rank on the inner
            # dimension. Gemma's published text configuration has 128 experts:
            # https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json
            # Layout: OpenPipe/ART src/art/utils/convert_moe_lora.py.
            is_fused_expert = name.endswith((
                ".moe.experts.lora_A.weight", ".moe.experts.base_layer.lora_A.weight"
            ))
            expected_inner = expected_rank * (packed_experts if is_fused_expert else 1)
            if shape[0] != expected_inner:
                raise ValueError("LoRA tensor rank differs from adapter configuration")
            pair_ranks[shape[0]] += 1
        else:
            partner = name.removesuffix(".lora_B.weight") + ".lora_A.weight"
            if partner not in shapes:
                raise ValueError("LoRA B tensor has no A partner")
    return {
        "tensor_count": len(shapes),
        "tensor_elements": elements,
        "dtype_counts": dict(dtypes),
        "all_values_finite": True,
        "nonzero_lora_b_elements": nonzero_b,
        "pair_inner_dimension_counts": dict(pair_ranks),
    }


def validate_adapter(directory: Path, require_trained: bool = False) -> dict[str, Any]:
    manifest = verified_manifest(directory)
    config = json.loads((directory / "adapter_config.json").read_text())
    label = manifest["label"]
    base = config.get("base_model_name_or_path")
    if label not in BASES or base not in BASES[label]:
        raise ValueError("adapter base is outside the selected model for this campaign label")
    if config.get("peft_type") != "LORA" or not isinstance(config.get("r"), int) or config["r"] <= 0:
        raise ValueError("adapter configuration has no valid LoRA rank")
    if isinstance(config["r"], bool):
        raise ValueError("LoRA rank must be an integer, not a boolean")
    if config.get("rank_pattern"):
        raise ValueError("per-layer rank patterns require separate validation")
    packed_experts = 128 if label == "gemma4" and config.get("art_lora_format") == "vllm" else 1
    summary = tensor_summary(directory / "adapter_model.safetensors", config["r"], packed_experts)
    step = manifest["checkpoint_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("invalid checkpoint step")
    if require_trained and (step == 0 or summary["nonzero_lora_b_elements"] == 0):
        raise ValueError("export is initial or has no nonzero LoRA B update")
    return {
        "label": label,
        "base_model": base,
        "checkpoint_step": step,
        "artifact_uri": manifest["artifact_uri"],
        "artifact_version": manifest["artifact_version"],
        "adapter_sha256": sha256_file(directory / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256_file(directory / "adapter_config.json"),
        "lora_rank": config["r"],
        "local_manifest_files_verified": True,
        "trained_export_checks_required": require_trained,
        **summary,
        "finality_proven": False,
        "limitations": [
            "Local file and tensor validation does not prove provider-job completion or serving compatibility.",
            "Nonzero finite tensors do not establish learning, response quality, or clinical safety.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-trained", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_adapter(args.directory, args.require_trained)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
