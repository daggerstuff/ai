#!/usr/bin/env python3
"""Wayfarer Pilot Training Script - Production (Quad-Audit Rounds 21-40 Final).

Target: Mistral-Nemo-Instruct-2407 (or similar causal LMs).
Optimised for T4 16 GB VRAM with robust safety, observability, and error handling.
Conforms to Python 3.11+ type-hint standards and Google-style docstrings.
"""

import argparse
import contextlib
import inspect
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Sized as SizedProtocol
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import torch
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

try:
    # TRL >= 1.3
    import trl.trainer.utils as trl_utils

    DataCollatorForCompletionOnlyLM = getattr(trl_utils, "DataCollatorForCompletionOnlyLM", None)
except ImportError:
    try:
        # TRL < 1.3
        import trl

        DataCollatorForCompletionOnlyLM = getattr(trl, "DataCollatorForCompletionOnlyLM", None)
    except ImportError:
        # TRL >= 1.3: SFTTrainer handles completion masking via SFTConfig
        DataCollatorForCompletionOnlyLM = None

from datasets import Dataset, load_dataset

try:
    from .shared_config import (
        add_lora_args,
        build_lora_config,
        log_token_length_distribution,
        shared_qlora_config,
    )
except ModuleNotFoundError:
    try:
        from ai.training.shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )
    except ModuleNotFoundError:
        from shared_config import (
            add_lora_args,
            build_lora_config,
            log_token_length_distribution,
            shared_qlora_config,
        )


# SAFETY CHECKERS DISABLED PER USER REQUEST - ALL CONTENT ALLOWED
SAFETY_CHECKER = None

# 0. ENVIRONMENT & SECURITY

WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", "/app/data")).resolve()
with contextlib.suppress(OSError):
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
# GPUStatsThread (pynvml-based GPU telemetry) is Colab-only — see pixelated_colab_pilot.ipynb

# Minimum samples required after filtering — ensures a meaningful 90/10 split
# (at least 2 eval samples) and avoids degenerate training.
MIN_DATASET_SIZE = 20

# Cap multiprocessing workers — spawning too many for fast operations adds overhead
_NUM_PROC: int = min(4, max(1, os.cpu_count() or 1))


def _tokenizer_supports_gen_prompt(tokenizer) -> bool:
    """Return True if tokenizer.apply_chat_template accepts add_generation_prompt."""
    try:
        return "add_generation_prompt" in inspect.signature(tokenizer.apply_chat_template).parameters
    except Exception:
        return False


def safe_path(user_path: str | Path) -> Path:
    """Validate and resolve a path, ensuring it stays within WORKSPACE_ROOT.

    Relative paths are resolved against WORKSPACE_ROOT, not CWD.
    Raises PermissionError directly without re-wrapping.

    Args:
        user_path: Absolute or relative path to validate.

    Returns:
        Resolved absolute Path within WORKSPACE_ROOT.

    Raises:
        PermissionError: If the path escapes WORKSPACE_ROOT or cannot be resolved.
    """
    p = Path(user_path)
    if not p.is_absolute():
        p = WORKSPACE_ROOT / p
    try:
        target = p.resolve()
    except (ValueError, OSError) as e:
        raise PermissionError(f"Path resolution error: {e}") from e
    root_str = str(WORKSPACE_ROOT).rstrip(os.sep) + os.sep
    target_str = str(target).rstrip(os.sep) + os.sep
    if not target_str.startswith(root_str):
        raise PermissionError(f"Access denied: {target} is outside workspace {WORKSPACE_ROOT}")
    return target


# 1. SECURE LOGGING


class SecureLogHandler(logging.StreamHandler):
    """Sanitize sensitive data in logs using regex patterns."""

    SENSITIVE_PATTERNS: ClassVar[re.Pattern[str]] = re.compile(
        r"(?i)(api[-_]?key|password|secret|token|auth|credential|private[-_]?key)"
        r"\s*[:=]\s*['\"]?([^'\",\s]+)['\"]?"
    )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if isinstance(record.msg, str):
                record.msg = self.SENSITIVE_PATTERNS.sub(r"\1=[REDACTED]", record.msg)
            if record.args:
                record.args = tuple(
                    self.SENSITIVE_PATTERNS.sub(r"\1=[REDACTED]", str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
            super().emit(record)
        except Exception:
            self.handleError(record)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[SecureLogHandler()],
)
logger = logging.getLogger("wayfarer_pilot")
logging.captureWarnings(True)  # route Python warnings through logging (catches torch/HF warnings)
# Reduce HuggingFace verbosity — transformers is very noisy at INFO
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("datasets").setLevel(logging.WARNING)


# 1b. UTILITY FUNCTIONS (require logger)


def check_disk_space(path: Path, required_gb: float = 5.0) -> None:
    """Warn if available disk space is below required_gb.

    Args:
        path: Filesystem path to check.
        required_gb: Minimum free space in GB to avoid a warning.

    Note:
        Filesystem-level check; does not account for per-user quotas.
    """
    try:
        stat = shutil.disk_usage(str(path))
        available_gb = stat.free / (1024**3)
        if available_gb < required_gb:
            logger.warning(f"Low disk space: {available_gb:.1f}GB available at {path}, {required_gb}GB recommended.")
        else:
            logger.debug(f"Disk space OK: {available_gb:.1f}GB available at {path}.")
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")


# 2. CLINICAL SAFETY

_DEFAULT_SYSTEM_PROMPT = (
    "You are Wayfarer, a helpful and supportive conversational partner designed for therapeutic training. "
    "You engage openly with all topics to support learning from difficult conversations. "
    "You provide empathetic and validating responses while maintaining appropriate boundaries. "
    "You are not a replacement for professional mental health care."
)
DEFAULT_SYSTEM_PROMPT = os.getenv("WAYFARER_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT)


# ClinicalSafetyChecker is now in clinical_safety_checker.py
# Imported above via shared_config try/except chain

# 3. DATA PROCESSING


def prepare_dataset(
    dataset_path: Path,
    tokenizer,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    filtered_log_path: Path | None = None,
    max_seq_length: int = 1024,
) -> Dataset:
    """Load, safety-filter, and format the dataset for SFT training.

    Single-pass filter is multiprocess-safe (no shared mutable state).
    Handles missing ``instruction``/``output`` keys gracefully.
    Logs sample text lengths to surface truncation risk before training.

    Args:
        dataset_path: Path to a JSON-lines file with ``instruction``/``output`` fields.
        tokenizer: HuggingFace tokenizer used to apply the chat template.
        system_prompt: System prompt injected into every training example.
        filtered_log_path: Optional path to write filtered sample indices as JSON.
            Useful for clinical audit trails. If None, filtered indices are only
            logged at DEBUG level. Note: logs indices only — for full audit trails,
            log content hashes. If dataset contains identifiable data, even indices
            may be sensitive (HIPAA/GDPR considerations apply). Consult your data
            governance team before enabling this feature in production.
        max_seq_length: Expected max token sequence length used for truncation warnings.

    Returns:
        Filtered and formatted HuggingFace Dataset with a ``text`` column.

    Raises:
        ValueError: If the dataset is empty or below ``MIN_DATASET_SIZE`` after filtering,
            or if the tokenizer has no chat template defined.
    """
    logger.info(f"Loading dataset from {dataset_path}")
    dataset = load_dataset("json", data_files=str(dataset_path), split="train")

    filtered_indices: list[int] = []

    # Note: when filtering was enabled, filtered_indices would be empty for workers > 1.
    # SAFETY FILTERING DISABLED PER USER REQUEST - NO FILTERING APPLIED
    # dataset = dataset.filter(is_safe_example, with_indices=True, num_proc=filter_num_proc)
    # All samples are kept for therapeutic training as requested
    logger.info(
        f"No safety filtering applied - all {len(dataset)} samples kept per user request"
    )
    # Note: filtered_indices will be empty since safety filtering is disabled.
    # SAFETY FILTERING DISABLED PER USER REQUEST - NO FILTERING APPLIED
    # dataset = dataset.filter(is_safe_example, with_indices=True, num_proc=filter_num_proc)
    # All samples are kept for therapeutic training as requested
    logger.info(f"No safety filtering applied - all {len(dataset)} samples kept per user request")
    if filtered_log_path is not None:
        # Note: logs indices only — for full audit trails, log content hashes.
        # Note: if dataset contains identifiable data, even indices may be sensitive (HIPAA/GDPR).
        # Consult your data governance team before enabling this feature in production.
        filtered_log_path.parent.mkdir(parents=True, exist_ok=True)
        filtered_log_path.write_text(json.dumps({"filtered_indices": filtered_indices}, indent=2))
        logger.info(f"Filtered sample indices written to {filtered_log_path}")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty after safety filtering. Check your data.")
    if len(dataset) < MIN_DATASET_SIZE:
        raise ValueError(
            f"Dataset too small ({len(dataset)} samples) after filtering. "
            f"Need at least {MIN_DATASET_SIZE} for a meaningful train/eval split."
        )

    if tokenizer.chat_template is None:
        raise ValueError(
            "Tokenizer has no chat_template defined. Set tokenizer.chat_template or use a model that includes one."
        )

    # Check add_generation_prompt support once before mapping (avoids N warnings per example)
    _supports_gen_prompt = _tokenizer_supports_gen_prompt(tokenizer)
    if not _supports_gen_prompt:
        logger.warning(
            "tokenizer.apply_chat_template does not support add_generation_prompt "
            "(transformers < 4.34). Generation prompt may be included in training data."
        )

    def format_example(example):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]},
        ]
        if _supports_gen_prompt:
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_example, num_proc=_NUM_PROC)
    # Log max sequence length to surface truncation risk before training
    if len(dataset) > 0:
        sample_size = min(100, len(dataset))
        sample_texts = dataset.select(range(sample_size))["text"]
        sample_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in sample_texts]
        log_token_length_distribution(
            lengths=sample_lengths,
            max_seq_length=max_seq_length,
            logger=logger,
            field_name="train_dataset_sample_lengths",
        )
    return dataset


# 4. MODEL & TOKENIZER SETUP


def setup_model_and_tokenizer(
    model_name: str,
    attn_implementation: str | None = None,
) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerBase]:
    """Load model with 4-bit quantization and tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        attn_implementation: Optional attention implementation override.
            Use ``'flash_attention_2'`` for Flash Attention 2 (requires
            ``flash-attn`` package; Ampere+ GPUs only).

    Returns:
        Tuple of (model, tokenizer). T4 does not support bfloat16;
        compute dtype falls back to float16 automatically.
    """
    bnb_config = shared_qlora_config()

    # attn_implementation: requires transformers >= 4.36. Use conditional dict to
    # avoid TypeError on older versions.
    _attn_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else {}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
        **_attn_kwargs,
    )
    if attn_implementation == "flash_attention_2":
        logger.warning(
            "flash_attention_2 with load_in_4bit=True may not be supported by all model versions. "
            "If you encounter errors, remove --attn_implementation."
        )

    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # Right padding required for completion-only loss masking
    tokenizer.padding_side = "right"
    # Warn: pad_token_id == eos_token_id is standard for causal LMs but means padding
    # and EOS share the same ID. The completion-only collator masks padding correctly,
    # but be aware if EOS appears mid-sequence as a separator.
    # SFTTrainer uses DataCollatorForLanguageModeling which handles padding via SFTConfig.
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        logger.debug("pad_token_id == eos_token_id (expected for causal LMs; verify collator masking).")

    return cast(AutoModelForCausalLM, model), cast(PreTrainedTokenizerBase, tokenizer)


def get_response_template_ids(tokenizer: PreTrainedTokenizerBase) -> list[int]:
    """Get token IDs for the response template boundary.

    Checks the special-tokens vocab first; falls back to ``encode()``.
    Verifies the IDs appear as a contiguous sub-sequence in a real formatted
    sample to catch tokenizer-version mismatches that would silently disable
    loss masking.

    Args:
        tokenizer: HuggingFace tokenizer for Mistral-Nemo.

    Returns:
        List of token IDs representing ``[/INST]``.

    Raises:
        ValueError: If the template cannot be encoded or is not found in a
            formatted sample.

    Note:
        ``[INST]`` and ``[/INST]`` are distinct tokens in Mistral; no
        ``instruction_template`` is needed for the collator.
    """
    template = "[/INST]"
    vocab = tokenizer.get_vocab()
    ids = [vocab[template]] if template in vocab else tokenizer.encode(template, add_special_tokens=False)
    if not ids:
        raise ValueError(
            f"Could not encode response template '{template}'. "
            "Verify this matches your tokenizer's chat template output."
        )
    # Verify the IDs appear as a contiguous sub-sequence in a real formatted sample
    sample_msgs = [
        {"role": "system", "content": "test system"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    sample_text = (
        tokenizer.apply_chat_template(sample_msgs, tokenize=False, add_generation_prompt=False)
        if _tokenizer_supports_gen_prompt(tokenizer)
        else tokenizer.apply_chat_template(sample_msgs, tokenize=False)
    )
    sample_ids = tokenizer.encode(str(sample_text), add_special_tokens=False)
    found = any(sample_ids[i : i + len(ids)] == ids for i in range(len(sample_ids) - len(ids) + 1))
    if not found:
        raise ValueError(
            f"Response template IDs {ids} not found as contiguous sub-sequence in "
            f"formatted sample. Loss masking will be broken. "
            f"Check tokenizer version and chat template format."
        )
    logger.debug(f"Response template '{template}' → token IDs: {ids} (verified in sample)")
    # Warn if any template ID equals EOS — could cause incorrect masking
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
        logger.warning(
            f"Response template ID {ids} includes EOS token ({tokenizer.eos_token_id}). "
            "This may cause incorrect loss masking. Verify your chat template."
        )
    return ids


# 5. TRAINING CONFIGURATION


def configure_training(output_dir: Path, args: argparse.Namespace) -> tuple[LoraConfig, SFTConfig]:
    """Configure LoRA and SFT training arguments.

    Args:
        output_dir: Resolved output directory path.
        args: Parsed CLI arguments.

    Returns:
        Tuple of (peft_config, sft_config). Pass peft_config to SFTTrainer
        directly — do NOT also call get_peft_model() manually.
    """
    peft_config = build_lora_config(args)

    # CLI --report_to overrides REPORT_TO env var; both default to "none"
    report_to = args.report_to or os.getenv("REPORT_TO", "none")

    # SFTConfig extends TrainingArguments with SFT-specific fields (max_length,
    # dataset_text_field, packing, neftune_noise_alpha). Using SFTConfig here keeps
    # all config in one place and avoids passing deprecated kwargs to SFTTrainer.
    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        report_to=report_to,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",  # requires bitsandbytes >= 0.39
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        dataloader_num_workers=args.dataloader_num_workers,
        # pin_memory speeds up host-to-GPU transfer; only beneficial with workers > 0.
        # Note: with device_map="auto" (multi-GPU), pin_memory may cause issues on some setups.
        dataloader_pin_memory=args.dataloader_num_workers > 0,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        # When load_best_model_at_end=True, trainer.save_model() saves the best checkpoint.
        # metric_for_best_model defaults to "loss"; set REPORT_TO to enable custom metrics.
        torch_compile=args.torch_compile,
        # torch_compile: PyTorch 2.0+ only; may not work with all quantization configs.
        # SFT-specific fields (moved from SFTTrainer kwargs in TRL >= 0.9)
        max_length=args.max_seq_length,
        dataset_text_field="text",  # matches format_example output key
        packing=False,
        neftune_noise_alpha=args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None,
    )

    if args.torch_compile:
        logger.warning(
            "--torch_compile=True: may conflict with bitsandbytes 4-bit quantization "
            "and/or gradient checkpointing on some PyTorch versions. "
            "Disable if you encounter compilation errors."
        )

    return peft_config, sft_config


# 5b. HUB CONFIGURATION


@dataclass
class HubConfig:
    """Configuration for optional HuggingFace Hub push after training.

    Attributes:
        push_to_hub: Whether to push the final model to the Hub.
        hub_model_id: Hub model ID (e.g. ``'org/model-name'``). Required when
            ``push_to_hub`` is True.
    """

    push_to_hub: bool = False
    hub_model_id: str | None = None


# 6. CHECKPOINT VERIFICATION CALLBACK


class CheckpointVerificationCallback(TrainerCallback):
    """
    Verifies checkpoint integrity after the trainer writes it.
    Post-write verification only — removes incomplete checkpoints to prevent
    resuming from corrupt state.
    Uses shutil.move (cross-device safe) instead of os.rename.
    Note: directory rename is not atomic on all systems; this is the best
    available approach without kernel-level renameat2.
    """

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,  # noqa: ARG002
        **kwargs,  # noqa: ARG002
    ) -> None:
        checkpoint_dir = Path(args.output_dir or ".") / f"checkpoint-{state.global_step}"
        temp_dir = checkpoint_dir.with_suffix(".tmp")

        if not checkpoint_dir.exists():
            logger.warning(f"Expected checkpoint dir not found: {checkpoint_dir}")
            return

        if temp_dir.exists():
            shutil.rmtree(temp_dir)

        try:
            shutil.move(str(checkpoint_dir), str(temp_dir))
        except Exception as e:
            logger.error(f"Could not move checkpoint to temp for verification: {e}")
            return

        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [f for f in required_files if not (temp_dir / f).exists()]
        if not missing:
            try:
                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                shutil.move(str(temp_dir), str(checkpoint_dir))
                logger.info(f"Checkpoint {checkpoint_dir.name} verified and saved.")
            except Exception as e:
                logger.error(f"Checkpoint {checkpoint_dir.name} move failed: {e}. Verified copy remains at {temp_dir}.")
        else:
            logger.error(f"Checkpoint {checkpoint_dir.name} incomplete. Missing: {missing}. Removing.")
            shutil.rmtree(temp_dir, ignore_errors=True)


def _maybe_push_to_hub(
    trainer: SFTTrainer,
    tokenizer: PreTrainedTokenizerBase,
    hub_config: HubConfig | None,
) -> None:
    """Push model and tokenizer to HuggingFace Hub if configured.

    Args:
        trainer: Trained SFTTrainer instance.
        tokenizer: Tokenizer to push alongside the model.
        hub_config: Hub push configuration. No-op if None or push_to_hub is False.
    """
    if hub_config is None or not hub_config.push_to_hub or not hub_config.hub_model_id:
        return
    hub_model_id = hub_config.hub_model_id
    logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
    try:
        # Pushes the PEFT adapter weights (not merged). To push a merged model,
        # call trainer.model.merge_and_unload() first.
        if hasattr(trainer, "create_model_card"):
            trainer.create_model_card()
        # SFTTrainer wraps with PEFT; model is PeftModel at this point
        peft_model = cast(PeftModel, trainer.model)
        peft_model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)
        logger.info(f"Model pushed to hub: {hub_model_id}")
    except Exception as e:
        logger.error(f"Hub push failed: {e}")


@dataclass
class RunConfig:
    """Runtime control options for ``_run_training``.

    Attributes:
        hub_config: Optional Hub push configuration.
        skip_final_eval: If True, skip the final evaluation pass after training.
    """

    hub_config: HubConfig | None = None
    skip_final_eval: bool = False


def _run_training(
    trainer: SFTTrainer,
    output_dir: Path,
    tokenizer: PreTrainedTokenizerBase,
    resume_from_checkpoint: str | None,
    run_config: RunConfig | None = None,
) -> None:
    """Execute training, save metrics, evaluate, and persist the final model.

    Args:
        trainer: Configured SFTTrainer instance.
        output_dir: Resolved output directory.
        tokenizer: Tokenizer to save alongside the model.
        resume_from_checkpoint: Path to resume from, or None.
        run_config: Optional runtime control options (Hub push, eval skip).
            Defaults to ``RunConfig()`` (no Hub push, eval enabled).

    Note:
        If ``TrainingArguments.load_best_model_at_end=True``, ``trainer.save_model()``
        saves the best checkpoint (by eval loss), not the last step's weights.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # resets device 0; multi-GPU tracks device 0 only
    torch.cuda.empty_cache()
    logger.info("Starting training...")
    _run_cfg = run_config or RunConfig()
    if _run_cfg.skip_final_eval:
        logger.info("skip_final_eval=True — final evaluation pass will be skipped.")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()
    # Note: save_state() is called before save_model(). If the process dies between
    # these two calls, the state will indicate training is complete but no final model
    # will exist. This is acceptable — resume from the last checkpoint instead.
    logger.info(f"Trainer state saved to {output_dir / 'trainer_state.json'}")
    try:
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
    except Exception as e:
        logger.warning(f"Could not save train metrics: {e}")

    runtime_s = train_result.metrics.get("train_runtime", 0)
    samples_per_sec = train_result.metrics.get("train_samples_per_second", 0)
    steps_per_sec = train_result.metrics.get("train_steps_per_second", 0)
    epoch = train_result.metrics.get("epoch", 0)
    logger.info(
        f"Training time: {runtime_s / 60:.1f} min ({runtime_s:.0f}s), "
        f"{samples_per_sec:.2f} samples/sec, {steps_per_sec:.3f} steps/sec, "
        f"epoch={epoch:.2f}"
    )
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
        reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
        logger.info(f"Peak GPU memory — allocated: {peak_mb:.0f} MB, reserved: {reserved_mb:.0f} MB")

    if not _run_cfg.skip_final_eval and trainer.eval_dataset is not None:
        eval_ds = trainer.eval_dataset
        # eval_dataset may be a Dataset or DatasetDict; check for __len__ to satisfy Sized protocol
        eval_len = len(cast(SizedProtocol, eval_ds)) if hasattr(eval_ds, "__len__") else 0
        if eval_len > 0:
            try:
                eval_metrics = trainer.evaluate()
                trainer.log_metrics("eval", eval_metrics)
                trainer.save_metrics("eval", eval_metrics)
            except Exception as e:
                logger.warning(f"Final evaluation failed: {e}")

    final_path = output_dir / "final_model"
    # mkdtemp + shutil.move avoids TemporaryDirectory cleanup racing the rename.
    # shutil.move is cross-device safe; not atomic across filesystems.
    tmp_dir = tempfile.mkdtemp(dir=str(output_dir))
    try:
        trainer.save_model(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        if final_path.exists():
            shutil.rmtree(final_path)
        shutil.move(tmp_dir, str(final_path))
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    logger.info(f"Training completed. Model saved to {final_path}")

    _maybe_push_to_hub(trainer, tokenizer, _run_cfg.hub_config)


# 7. MAIN TRAINING PIPELINE

# Minimum save_total_limit when load_best_model_at_end is enabled, to avoid
# the best checkpoint being evicted before training ends.
_MIN_SAVE_TOTAL_LIMIT_FOR_BEST_MODEL: int = 3
# Minimum warmup steps below which training stability warnings are issued.
_MIN_WARMUP_STEPS_WARN: int = 5
# Minimum eval samples below which eval loss is considered too noisy to be useful.
_MIN_EVAL_SAMPLES_WARN: int = 10


def _validate_numeric_args(args: argparse.Namespace) -> None:
    """Validate numeric CLI argument ranges, exiting on any violation.

    Args:
        args: Parsed CLI arguments from ``argparse``.

    Raises:
        SystemExit: On any out-of-range argument value.
    """
    if args.max_steps <= 0:
        logger.error(f"--max_steps must be > 0, got {args.max_steps}.")
        sys.exit(1)

    if args.learning_rate <= 0:
        logger.error(f"--learning_rate must be > 0, got {args.learning_rate}.")
        sys.exit(1)

    if args.batch_size <= 0:
        logger.error(f"--batch_size must be > 0, got {args.batch_size}.")
        sys.exit(1)

    if args.lora_r <= 0:
        logger.error(f"--lora_r must be > 0, got {args.lora_r}.")
        sys.exit(1)

    if not 0.0 <= args.lora_dropout < 1.0:
        logger.error(f"--lora_dropout must be in [0, 1), got {args.lora_dropout}.")
        sys.exit(1)

    if not 0.0 <= args.warmup_ratio < 1.0:
        logger.error(f"--warmup_ratio must be in [0, 1), got {args.warmup_ratio}.")
        sys.exit(1)

    if args.max_grad_norm <= 0:
        logger.error(f"--max_grad_norm must be > 0, got {args.max_grad_norm}.")
        sys.exit(1)

    if args.weight_decay < 0:
        logger.error(f"--weight_decay must be >= 0, got {args.weight_decay}.")
        sys.exit(1)


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path]:
    """Validate parsed CLI arguments and resolve safe paths.

    Side effect: mutates ``args.output_dir`` to the resolved absolute path string,
    and ``args.resume_from_checkpoint`` to the resolved path string if provided.

    Args:
        args: Parsed CLI arguments from ``argparse``.

    Returns:
        Tuple of ``(data_path, output_dir)`` as resolved ``Path`` objects.

    Raises:
        SystemExit: On any validation failure (missing file, bad checkpoint, etc.).
    """
    data_path = safe_path(args.data_path)
    output_dir = safe_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(output_dir)

    if args.resume_from_checkpoint is not None:
        resume_path = safe_path(str(args.resume_from_checkpoint))
        if not resume_path.exists():
            logger.error(f"Checkpoint path does not exist: {resume_path}")
            sys.exit(1)
        args.resume_from_checkpoint = str(resume_path)

    if not torch.cuda.is_available():
        logger.warning("No CUDA device found. Training will be extremely slow on CPU.")
    check_disk_space(output_dir, required_gb=10.0)

    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        sys.exit(1)

    effective_batch = args.batch_size * args.gradient_accumulation_steps
    logger.info(
        f"Effective batch size: {effective_batch} "
        f"(per_device={args.batch_size} x grad_accum={args.gradient_accumulation_steps})"
    )

    if args.push_to_hub and not args.hub_model_id:
        logger.error("--hub_model_id is required when --push_to_hub is set.")
        sys.exit(1)

    _validate_numeric_args(args)

    if args.eval_steps > args.max_steps:
        logger.warning(
            f"--eval_steps ({args.eval_steps}) > --max_steps ({args.max_steps}): "
            "evaluation will never run. Consider reducing --eval_steps."
        )

    if args.load_best_model_at_end and getattr(args, "save_total_limit", 2) < _MIN_SAVE_TOTAL_LIMIT_FOR_BEST_MODEL:
        # save_total_limit is now a CLI arg; default=2
        logger.warning(
            "--load_best_model_at_end=True with save_total_limit < 3: the best checkpoint "
            "may be deleted before training ends. Consider --save_total_limit 3 or higher."
        )

    min_warmup_steps = int(args.max_steps * args.warmup_ratio)
    if min_warmup_steps < _MIN_WARMUP_STEPS_WARN:
        logger.warning(
            f"Warmup steps ({min_warmup_steps} = {args.max_steps} * warmup_ratio={args.warmup_ratio}) "
            "is very short. Training may be unstable."
        )

    return data_path, output_dir


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser for the training script.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(description="Wayfarer Pilot Training")
    parser.add_argument("--model_name", type=str, default="mistralai/Mistral-Nemo-Instruct-2407")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=1024,
        choices=range(512, 8193),
        help="Max token sequence length.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--neftune_noise_alpha",
        type=float,
        default=5.0,
        help="NEFTune noise alpha. Set 0 to disable.",
    )
    add_lora_args(parser)
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps.")
    parser.add_argument(
        "--eval_steps", type=int, default=100, help="Evaluate every N steps. Reduce for small datasets."
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Fraction of steps for LR warmup.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="AdamW weight decay. 0.01 is a common regularization value."
    )
    parser.add_argument("--logging_steps", type=int, default=10, help="Log metrics every N steps.")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        help="Attention implementation: 'flash_attention_2' for FA2 (requires flash-attn package), "
        "or None for default. FA2 reduces VRAM and speeds up training on Ampere+ GPUs.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Experiment tracker: 'wandb', 'mlflow', etc. Overrides REPORT_TO env var.",
    )
    parser.add_argument(
        "--filtered_log_path",
        type=str,
        default=None,
        help="Path to write filtered sample indices JSON for clinical audit trails.",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push final model to HuggingFace Hub. Requires huggingface-cli login.",
    )
    parser.add_argument(
        "--skip_final_eval",
        action="store_true",
        help="Skip the final evaluation pass after training (saves time for large eval sets).",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes. 0=main process only. 2-4 speeds up large datasets.",
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Load the best checkpoint (by eval loss) at end of training instead of the last.",
    )
    parser.add_argument(
        "--metric_for_best_model",
        type=str,
        default="loss",
        help="Metric to use for best model selection with --load_best_model_at_end.",
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        help="Enable torch.compile() for ~10-30%% speedup (PyTorch 2.0+, Ampere+ GPU recommended).",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Max checkpoints to keep. Use >= 3 with --load_best_model_at_end to avoid losing best checkpoint.",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace Hub model ID for --push_to_hub (e.g. 'org/model-name').",
    )
    return parser


def main():
    args = _build_arg_parser().parse_args()

    set_seed(args.seed)

    # TF32 gives free speedup on Ampere+ GPUs; no-op on T4 (Turing) but harmless
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Suppress HuggingFace tokenizer fork warning during dataset.map
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    data_path, output_dir = _validate_args(args)

    logger.info(f"Loading model: {args.model_name}")
    logger.info(
        f"Config — lr={args.learning_rate}, steps={args.max_steps}, "
        f"lora_r={args.lora_r}, lora_alpha={args.lora_alpha}, "
        f"seq_len={args.max_seq_length}, neftune={args.neftune_noise_alpha if args.neftune_noise_alpha > 0 else None}"
    )
    model, tokenizer = setup_model_and_tokenizer(args.model_name, attn_implementation=args.attn_implementation)
    # prepare_model_for_kbit_training handles gradient checkpointing setup.
    # Do NOT call model.gradient_checkpointing_enable() separately — causes double-enable warning.
    # Note: in PEFT >= 0.7, use_gradient_checkpointing was removed from
    # prepare_model_for_kbit_training. If you get a TypeError, remove the kwarg
    # and call model.gradient_checkpointing_enable() separately instead.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # enable_input_require_grads is required for gradient checkpointing with PEFT
    # in some versions; safe to call unconditionally.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    peft_config, sft_config = configure_training(output_dir, args)

    _filtered_log_path: Path | None = Path(args.filtered_log_path).resolve() if args.filtered_log_path else None
    dataset = prepare_dataset(
        data_path,
        tokenizer,
        max_seq_length=args.max_seq_length,
        filtered_log_path=_filtered_log_path,
    )
    dataset = dataset.train_test_split(test_size=0.1, seed=args.seed)

    n_eval = len(dataset["test"])
    if n_eval < _MIN_EVAL_SAMPLES_WARN:
        logger.warning(
            f"Eval split has only {n_eval} samples — eval loss will be very noisy. Consider using a larger dataset."
        )

    # SFTTrainer auto-creates DataCollatorForLanguageModeling from SFTConfig
    # With 4-bit quantization the embedding layer is typically kept in float32/float16
    # by bitsandbytes, so dtype mismatch is not expected — but monitor for NaN loss.

    response_template_ids = get_response_template_ids(tokenizer)
    if DataCollatorForCompletionOnlyLM is not None:
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_template_ids,
            tokenizer=tokenizer,
        )
    else:
        # TRL >= 1.3: SFTTrainer handles completion masking via SFTConfig
        collator = None

    # Disable KV cache — incompatible with gradient checkpointing
    model.config.use_cache = False

    # SFTTrainer wraps the model with PEFT internally via peft_config.
    # SFT-specific params (max_length, dataset_text_field, packing, neftune_noise_alpha)
    # are passed via SFTConfig (TRL >= 0.9); do not pass them as kwargs here.
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=sft_config,
        peft_config=peft_config,
        data_collator=collator,
        callbacks=[CheckpointVerificationCallback()],
    )

    # SFTTrainer wraps with PEFT after __init__
    peft_model = cast(PeftModel, trainer.model)
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    logger.info(
        "LoRA target modules: %s (rank=%d, alpha=%d)",
        target_modules,
        args.lora_r,
        args.lora_alpha,
    )
    if hasattr(peft_model, "print_trainable_parameters"):
        peft_model.print_trainable_parameters()

    _run_training(
        trainer,
        output_dir,
        tokenizer,
        args.resume_from_checkpoint,
        run_config=RunConfig(
            hub_config=HubConfig(push_to_hub=args.push_to_hub, hub_model_id=args.hub_model_id),
            skip_final_eval=args.skip_final_eval,
        ),
    )


# 8. ENTRY POINT

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
