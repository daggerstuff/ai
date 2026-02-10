#!/usr/bin/env python3
"""
Pixelated Empathy Model Training Script (Phase 3.1)
Supports Unsloth (optimized) and HuggingFace (standard) training pipelines.

Usage:
    uv run python ai/training/train_pixel.py --config ai/training/ready_packages/configs/hyperparameters/enhanced_training_config.json
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# Try to import Unsloth for optimization
try:
    from unsloth import FastLanguageModel

    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PixelTrainer")


class PixelTrainer:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.model_name = self.config.get("base_model", "LatitudeGames/Wayfarer-2-12B")
        self.output_dir = self.config.get("training_parameters", {}).get("output_dir", "./output")
        self.device_map = "auto"

        # Load component config for weights
        self.components = self.config.get("component_specific_config", {})

        logger.info(f"Initialized PixelTrainer with model: {self.model_name}")
        logger.info(f"Unsloth optimization available: {UNSLOTH_AVAILABLE}")

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load and validate training configuration."""
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r") as f:
            config = json.load(f)

        return config

    def setup_model_and_tokenizer(self) -> Tuple[Any, AutoTokenizer]:
        """Load model and tokenizer using Unsloth if available, else HF. Handles CPU fallback."""

        max_seq_length = self.config.get("context_config", {}).get("training_max_length", 2048)
        dtype = None  # Auto detection

        use_cuda = torch.cuda.is_available()
        # Only use 4bit quantization if CUDA is available
        load_in_4bit = use_cuda
        device_map = "auto" if use_cuda else "cpu"

        if UNSLOTH_AVAILABLE and use_cuda:
            logger.info("⚡ Loading model via Unsloth (GPU)...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=max_seq_length,
                dtype=dtype,
                load_in_4bit=load_in_4bit,
                device_map=device_map,
            )

            # Configure LoRA via Unsloth
            lora_conf = self.config.get("lora_config", {})
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_conf.get("lora_r", 16),
                target_modules=lora_conf.get(
                    "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
                lora_alpha=lora_conf.get("lora_alpha", 16),
                lora_dropout=lora_conf.get(
                    "lora_dropout", 0
                ),  # Unsloth handles dropout differently, usually 0
                bias=lora_conf.get("lora_bias", "none"),
                use_gradient_checkpointing="unsloth",  # Use Unsloth's checkpointing
                random_state=3407,
                use_rslora=False,
                loftq_config=None,
            )
        else:
            mode_str = "Standard HuggingFace (GPU)" if use_cuda else "Standard HuggingFace (CPU)"
            logger.info(f"🐢 Loading model via {mode_str}...")

            quantization_config = None
            if use_cuda:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map=device_map,
                torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
                if use_cuda and torch.cuda.get_device_capability()[0] >= 8
                else None,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            tokenizer.padding_side = "right"  # Fix weird overflow issue with clean llama2/mistral

            # Configure LoRA via PEFT
            lora_conf = self.config.get("lora_config", {})
            peft_config = LoraConfig(
                r=lora_conf.get("lora_r", 16),
                lora_alpha=lora_conf.get("lora_alpha", 16),
                lora_dropout=lora_conf.get("lora_dropout", 0.05),
                bias=lora_conf.get("lora_bias", "none"),
                task_type="CAUSAL_LM",
                target_modules=lora_conf.get(
                    "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
                ),
            )
            model = get_peft_model(model, peft_config)

        # Tokenizer setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    def load_data(self, tokenizer: AutoTokenizer) -> DatasetDict:
        """Load and preprocess datasets mapping to components."""
        data_config = self.config.get("dataset_config", {})
        main_file = data_config.get("ultimate_final_dataset")

        # In a real scenario, this would handle complex merging based on components
        # For Phase 3.1, we assume a pre-consolidated JSONL file

        data_path = Path(main_file) if main_file and Path(main_file).exists() else None

        if not data_path:
            logger.warning(
                f"Main dataset {main_file} not found. Using dummy data for pipeline verification."
            )
            # Create dummy data for pipeline validation
            dummy_data = [
                {
                    "conversations": [
                        {"role": "user", "content": "I feel sad."},
                        {"role": "assistant", "content": "I hear you."},
                    ]
                }
            ] * 10
            dataset = Dataset.from_list(dummy_data)
        else:
            logger.info(f"Loading dataset from {data_path}")
            dataset = load_dataset("json", data_files=str(data_path), split="train")

        # Standard ChatML formatting function
        def formatting_func(examples):
            convs = examples["conversations"]
            texts = []
            for conv in convs:
                # Basic ChatML handling
                formatted = tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=False
                )
                texts.append(formatted)
            return {"text": texts}

        dataset = dataset.map(formatting_func, batched=True)

        # Split logic checks config
        train_split = data_config.get("train_split", 0.9)
        val_split = data_config.get("val_split", 0.1)
        # Normalize ratios
        total = train_split + val_split
        train_ratio = train_split / total
        val_ratio = val_split / total

        # First split: train vs rest
        splits = dataset.train_test_split(test_size=val_ratio, seed=42)

        return splits

    def train(self):
        """Execute the training loop."""
        model, tokenizer = self.setup_model_and_tokenizer()
        dataset = self.load_data(tokenizer)

        params = self.config.get("training_parameters", {})
        h100_opts = self.config.get("h100_optimizations", {})

        args = TrainingArguments(
            output_dir=self.output_dir,
            per_device_train_batch_size=params.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=params.get("gradient_accumulation_steps", 4),
            warmup_steps=params.get("warmup_steps", 100),
            max_steps=params.get("max_steps", -1),
            num_train_epochs=params.get("num_train_epochs", 1),
            learning_rate=params.get("learning_rate", 2e-4),
            fp16=not h100_opts.get("bf16", False),
            bf16=h100_opts.get("bf16", False),
            logging_steps=self.config.get("logging", {}).get("logging_steps", 10),
            optim=(
                "adamw_8bit"
                if UNSLOTH_AVAILABLE and torch.cuda.is_available()
                else "paged_adamw_8bit"
                if torch.cuda.is_available()
                else "adamw_torch"
            ),
            weight_decay=params.get("weight_decay", 0.01),
            lr_scheduler_type="linear",
            seed=3407,
            report_to="none",  # Change to wandb in production
        )

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            args=args,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )

        logger.info("🚀 Starting training...")
        if UNSLOTH_AVAILABLE:
            # Unsloth specific optimizations if needed
            pass

        trainer_stats = trainer.train()
        logger.info(f"Training complete: {trainer_stats}")

        # Save model
        logger.info("Saving model...")
        model.save_pretrained(os.path.join(self.output_dir, "lora_model"))
        tokenizer.save_pretrained(os.path.join(self.output_dir, "lora_model"))

        # Merge if requested (optional logic here)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()

    trainer = PixelTrainer(args.config)
    trainer.train()
