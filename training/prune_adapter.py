#!/usr/bin/env python3
"""Magnitude-based LoRA adapter pruning (PIX-4345 Appendix E Step 6).

Applies L1-unstructured pruning to LoRA adapter weights (not the base model),
then optionally runs a 1-epoch recovery fine-tune on a subset of the training
data to restore quality.  Finally, verifies domain score remains within 5 %
of the pre-prune benchmark.

Blueprint ref:
  - Appendix C  (pruning schedule example)
  - Appendix E  Step 6 (pruning, pre-deployment)
  - §2          (cost optimization: pruning + distillation)

Usage (real GPU)::

    python -m ai.training.prune_adapter \
        --adapter-path ./lora-out \
        --amount 0.3 \
        --recovery-data ./data/curated/sft_chatml/train.jsonl \
        --recovery-frac 0.1 \
        --base-model deepseek-ai/DeepSeek-V4-Pro \
        --benchmark-pre ./benchmarks/pre_train_2026-08-01.json

Usage (mock / dry-run, no GPU)::

    python -m ai.training.prune_adapter \
        --adapter-path ./lora-out \
        --amount 0.3 \
        --dry-run

Target: 30-50 % fewer active params with < 5 % quality loss post-recovery.
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

# Quality gate: domain score must stay within this fraction of pre-prune score.
QUALITY_LOSS_THRESHOLD = 0.05  # < 5 %


@dataclass
class PruningStats:
    """Summary of a pruning run."""

    adapter_path: str
    amount: float  # pruning fraction (0.0-1.0)
    total_params: int = 0
    pruned_params: int = 0
    sparsity: float = 0.0  # pruned / total
    target_modules: list[str] = field(default_factory=list)
    recovery_epochs: int = 0
    recovery_data_fraction: float = 0.0
    pre_prune_domain_score: float | None = None
    post_prune_domain_score: float | None = None
    quality_loss: float | None = None
    quality_gate_passed: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core pruning logic
# ---------------------------------------------------------------------------

def _iter_lora_linear_modules(model: Any) -> list[tuple[str, Any]]:
    """Yield (module_fqn, module) for every LoRA linear layer in the model.

    Works with both PEFT LoraModel wrappers and plain nn.Module trees where
    LoRA layers are identified by the ``lora_A`` / ``lora_B`` attribute pattern.
    """
    import torch.nn as nn

    modules: list[tuple[str, Any]] = []

    for name, module in model.named_modules():
        # PEFT LoRA layers expose ``lora_A`` and ``lora_B`` as ModuleDict.
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            modules.append((name, module))
        # Fallback: detect by class name for non-PEFT LoRA implementations.
        elif isinstance(module, nn.Linear) and (
            hasattr(module, "lora_A") or "lora" in name.lower()
        ):
            modules.append((name, module))

    return modules


def prune_lora_adapter(
    model: Any,
    amount: float = 0.3,
    target_modules: list[str] | None = None,
) -> PruningStats:
    """Apply L1-unstructured pruning to LoRA weights in-place.

    Parameters
    ----------
    model
        A PEFT model (or plain ``nn.Module``) with LoRA adapters loaded.
    amount
        Fraction of weights to prune (0.0-1.0).  0.3 = 30 %.
    target_modules
        Optional list of module FQNs to prune.  If ``None``, all LoRA layers
        are pruned.

    Returns
    -------
    PruningStats with total/pruned param counts and sparsity.
    """
    import torch.nn.utils.prune as prune

    lora_modules = _iter_lora_linear_modules(model)
    if target_modules:
        target_set = set(target_modules)
        lora_modules = [(n, m) for n, m in lora_modules if n in target_set]

    if not lora_modules:
        logger.warning("No LoRA modules found for pruning.")
        return PruningStats(
            adapter_path="",
            amount=amount,
            target_modules=[],
        )

    total_params = 0
    pruned_params = 0
    pruned_module_names: list[str] = []

    for fqn, module in lora_modules:
        # Prune lora_A weights (the down-projection; pruning here reduces
        # active rank effectively).  Some implementations store lora_A as
        # a ModuleDict of nn.Linear per adapter name.
        lora_a = getattr(module, "lora_A", None)
        lora_b = getattr(module, "lora_B", None)

        counters: dict[str, list[int]] = {"total": [total_params], "pruned": [pruned_params]}

        for lora_dict in (lora_a, lora_b):
            if lora_dict is None:
                continue
            # PEFT stores as ModuleDict: {adapter_name: nn.Linear}
            if hasattr(lora_dict, "items"):
                for _adapter_name, lin in lora_dict.items():
                    _prune_linear(lin, amount, fqn, total_params_ref=counters["total"], pruned_ref=counters["pruned"])
                    pruned_module_names.append(fqn)
            elif hasattr(lora_dict, "weight"):
                _prune_linear(lora_dict, amount, fqn, total_params_ref=counters["total"], pruned_ref=counters["pruned"])
                pruned_module_names.append(fqn)

        total_params = counters["total"][0]
        pruned_params = counters["pruned"][0]

    sparsity = pruned_params / total_params if total_params > 0 else 0.0

    stats = PruningStats(
        adapter_path="",
        amount=amount,
        total_params=total_params,
        pruned_params=pruned_params,
        sparsity=round(sparsity, 4),
        target_modules=sorted(set(pruned_module_names)),
    )
    logger.info(
        "Pruned %d / %d params (%.1f%% sparsity) across %d modules",
        pruned_params,
        total_params,
        sparsity * 100,
        len(set(pruned_module_names)),
    )
    return stats


def _prune_linear(
    lin: Any,
    amount: float,
    fqn: str,
    total_params_ref: list[int],
    pruned_ref: list[int],
) -> None:
    """Prune a single nn.Linear's weight tensor and make it permanent."""
    import torch.nn.utils.prune as prune

    weight = lin.weight
    n_params = weight.numel()
    total_params_ref[0] += n_params

    # Apply L1 unstructured pruning.
    prune.l1_unstructured(lin, name="weight", amount=amount)

    # Count zeros = pruned params.
    zero_count = int((lin.weight == 0).sum().item())
    pruned_ref[0] += zero_count

    # Make pruning permanent (remove the forward_pre_hook).
    prune.remove(lin, "weight")

    logger.debug("Pruned %s: %d / %d zeros (%.1f%%)", fqn, zero_count, n_params, zero_count / max(n_params, 1) * 100)


# ---------------------------------------------------------------------------
# Recovery fine-tune
# ---------------------------------------------------------------------------

def recovery_finetune(
    model: Any,
    tokenizer: Any,
    data_path: Path,
    fraction: float = 0.1,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    batch_size: int = 2,
    grad_accum_steps: int = 8,
) -> dict[str, Any]:
    """Run a short recovery fine-tune on a fraction of the training data.

    Uses the same QLoRA + LoRA config as the original SFT run but with a
    lower learning rate and fewer epochs.  This restores quality lost to
    pruning by re-training the surviving weights.

    Returns a dict with training metrics (loss history, steps).
    """
    import json as _json

    from transformers import TrainingArguments

    # Load and subsample the dataset.
    records: list[dict[str, Any]] = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_json.loads(line))

    n_subset = max(1, int(len(records) * fraction))
    subset = records[:n_subset]
    logger.info("Recovery fine-tune: %d samples (%.0f%% of %d)", n_subset, fraction * 100, len(records))

    # Build a simple text dataset from the JSONL records.
    import torch
    from torch.utils.data import Dataset

    class _RecoveryDataset(Dataset):
        def __init__(self, samples: list[dict[str, Any]], tok: Any, max_len: int = 2048) -> None:
            self.samples = samples
            self.tok = tok
            self.max_len = max_len

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            sample = self.samples[idx]
            messages = sample.get("messages", [])
            text = self.tok.apply_chat_template(messages, tokenize=False) if hasattr(self.tok, "apply_chat_template") else _json.dumps(messages)
            enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
            return {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": enc["input_ids"].squeeze(0),
            }

    ds = _RecoveryDataset(subset, tokenizer)

    # Use Trainer for recovery.
    from transformers import Trainer

    output_dir = Path("./prune-recovery-out")
    output_dir.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        bf16=True,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
    )

    train_result = trainer.train()
    metrics = {
        "train_loss": train_result.training_loss,
        "epochs": epochs,
        "samples": n_subset,
        "fraction": fraction,
        "global_step": train_result.global_step,
    }
    logger.info("Recovery fine-tune complete: %s", metrics)
    return metrics


# ---------------------------------------------------------------------------
# Quality verification
# ---------------------------------------------------------------------------

def verify_quality(
    pre_prune_report: Path,
    post_prune_report: Path,
) -> dict[str, Any]:
    """Compare pre/post-prune benchmark reports and check the quality gate.

    The domain score must not drop by more than ``QUALITY_LOSS_THRESHOLD``
    (5 %) relative to the pre-prune score.
    """
    pre_data = json.loads(pre_prune_report.read_text(encoding="utf-8"))
    post_data = json.loads(post_prune_report.read_text(encoding="utf-8"))

    pre_domain = _extract_domain_score(pre_data)
    post_domain = _extract_domain_score(post_data)

    if pre_domain is None or post_domain is None:
        return {
            "pre_prune_domain_score": pre_domain,
            "post_prune_domain_score": post_domain,
            "quality_loss": None,
            "quality_gate_passed": None,
            "verdict": "INSUFFICIENT_DATA (domain score missing from one or both reports)",
        }

    quality_loss = (pre_domain - post_domain) / pre_domain if pre_domain > 0 else 0.0
    quality_loss = max(quality_loss, 0.0)
    gate_passed = quality_loss <= QUALITY_LOSS_THRESHOLD

    if gate_passed:
        verdict = f"PASS (quality loss {quality_loss:.1%} ≤ {QUALITY_LOSS_THRESHOLD:.0%} threshold)"
    else:
        verdict = f"FAIL (quality loss {quality_loss:.1%} > {QUALITY_LOSS_THRESHOLD:.0%} threshold)"

    return {
        "pre_prune_domain_score": round(pre_domain, 4),
        "post_prune_domain_score": round(post_domain, 4),
        "quality_loss": round(quality_loss, 4),
        "quality_gate_passed": gate_passed,
        "verdict": verdict,
    }


def _extract_domain_score(report: dict[str, Any]) -> float | None:
    """Extract the domain benchmark score from a benchmark report."""
    for result in report.get("results", []):
        if result.get("category") == "domain":
            return result.get("score")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Magnitude-based LoRA adapter pruning (PIX-4345 Appendix E Step 6)",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to the LoRA adapter directory (e.g. ./lora-out)",
    )
    parser.add_argument(
        "--amount",
        type=float,
        default=0.3,
        help="Fraction of LoRA weights to prune (0.0-1.0). Default: 0.3 (30%%)",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model name or HF path (required for non-dry-run)",
    )
    parser.add_argument(
        "--recovery-data",
        type=str,
        default=None,
        help="Path to training JSONL for recovery fine-tune",
    )
    parser.add_argument(
        "--recovery-frac",
        type=float,
        default=0.1,
        help="Fraction of training data to use for recovery. Default: 0.1 (10%%)",
    )
    parser.add_argument(
        "--recovery-epochs",
        type=int,
        default=1,
        help="Number of recovery fine-tune epochs. Default: 1",
    )
    parser.add_argument(
        "--benchmark-pre",
        type=str,
        default=None,
        help="Path to pre-prune benchmark report JSON",
    )
    parser.add_argument(
        "--benchmark-post",
        type=str,
        default=None,
        help="Path to post-prune benchmark report JSON (for quality verification)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="ai/training/pruning",
        help="Directory for pruning reports",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip model loading; only emit a plan and estimated stats",
    )
    parser.add_argument(
        "--save-adapter",
        type=str,
        default=None,
        help="Path to save the pruned (and optionally recovered) adapter",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stats = PruningStats(
            adapter_path=args.adapter_path,
            amount=args.amount,
            target_modules=["(dry-run: all LoRA layers)"],
            recovery_epochs=args.recovery_epochs if args.recovery_data else 0,
            recovery_data_fraction=args.recovery_frac if args.recovery_data else 0.0,
        )
        report_path = out_dir / f"prune_plan_{Path(args.adapter_path).name}.json"
        report_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
        print(f"[prune] dry-run plan written to {report_path}")
        print(f"[prune] amount={args.amount} recovery={'yes' if args.recovery_data else 'no'}")
        return 0

    if not args.base_model:
        print("[prune] ERROR: --base-model is required for non-dry-run mode")
        return 1

    # Load model + adapter.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model = model.merge_and_unload() if hasattr(model, "merge_and_unload") else model

    # Prune.
    logger.info("Pruning LoRA adapter (amount=%.2f)", args.amount)
    stats = prune_lora_adapter(model, amount=args.amount)
    stats.adapter_path = args.adapter_path

    # Recovery fine-tune.
    if args.recovery_data:
        logger.info("Running recovery fine-tune (%d epochs, %.0f%% of data)", args.recovery_epochs, args.recovery_frac * 100)
        recovery_metrics = recovery_finetune(
            model=model,
            tokenizer=tokenizer,
            data_path=Path(args.recovery_data),
            fraction=args.recovery_frac,
            epochs=args.recovery_epochs,
        )
        stats.recovery_epochs = args.recovery_epochs
        stats.recovery_data_fraction = args.recovery_frac
    else:
        recovery_metrics = None

    # Save pruned adapter.
    if args.save_adapter:
        save_path = Path(args.save_adapter)
        save_path.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            logger.info("Saved pruned adapter to %s", save_path)

    # Quality verification.
    quality_report: dict[str, Any] | None = None
    if args.benchmark_pre and args.benchmark_post:
        quality_report = verify_quality(
            Path(args.benchmark_pre),
            Path(args.benchmark_post),
        )
        stats.pre_prune_domain_score = quality_report.get("pre_prune_domain_score")
        stats.post_prune_domain_score = quality_report.get("post_prune_domain_score")
        stats.quality_loss = quality_report.get("quality_loss")
        stats.quality_gate_passed = quality_report.get("quality_gate_passed")

    # Write report.
    report: dict[str, Any] = {
        "pruning_stats": stats.to_dict(),
        "recovery_metrics": recovery_metrics,
        "quality_report": quality_report,
    }
    report_path = out_dir / f"prune_report_{Path(args.adapter_path).name}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[prune] report written to {report_path}")
    print(f"[prune] sparsity={stats.sparsity:.1%} pruned={stats.pruned_params}/{stats.total_params}")

    if quality_report:
        print(f"[prune] quality: {quality_report['verdict']}")

    # Exit non-zero if quality gate failed.
    if stats.quality_gate_passed is False:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
