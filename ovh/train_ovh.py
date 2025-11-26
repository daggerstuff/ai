#!/usr/bin/env python3
"""
OVH AI Training - Wayfarer-2-12B Supervised Fine-Tuning

Adapted from:
- ai/pipelines/wayfarer_supervised.py (ChatML pipeline)
- ai/lightning/train_optimized.py (H100 optimizations)

Supports:
- Staged training (Foundation → Reasoning → Voice)
- MoE architecture with LoRA fine-tuning
- OVH Object Storage for data and checkpoints
- WandB integration for experiment tracking

Dataset Structure (on OVH Object Storage):
  /data/
  ├── acquired/               # CoT reasoning + mental health counseling
  │   ├── cot_reasoning.json
  │   └── mental_health_counseling.json
  ├── lightning/              # Expert-specific training data
  │   ├── expert_*.json
  │   └── train.json
  ├── voice/                  # Tim Fletcher voice data
  ├── gdrive/                 # Google Drive datasets
  └── config/
      ├── dataset_registry.json
      └── moe_training_config.json
"""

import os
import sys
import json
import glob
import random
import signal
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("train_ovh")

# ===========================================
# Configuration
# ===========================================

DEFAULT_CONFIG = {
    # Paths (OVH volumes)
    "data_dir": "/data",
    "checkpoint_dir": "/checkpoints",

    # Base model
    "base_model": "LatitudeGames/Wayfarer-2-12B",

    # Training stages
    "training_stages": {
        "foundation": {
            "datasets": ["acquired/mental_health_counseling.json", "lightning/train.json"],
            "epochs": 3,
            "learning_rate": 2e-4,
            "description": "Natural therapeutic dialogue patterns"
        },
        "reasoning": {
            "datasets": ["acquired/cot_reasoning.json"],
            "epochs": 2,
            "learning_rate": 1e-4,
            "description": "Clinical reasoning patterns (CoT)"
        },
        "voice": {
            "datasets": ["voice/synthetic_conversations.json"],
            "epochs": 2,
            "learning_rate": 5e-5,
            "description": "Tim Fletcher teaching style"
        }
    },

    # MoE Configuration
    "moe": {
        "num_experts": 4,
        "expert_domains": ["psychology", "mental_health", "bias_detection", "general_therapeutic"],
        "expert_capacity": 2,
        "load_balancing_weight": 0.01
    },

    # LoRA Configuration
    "lora": {
        "r": 16,
        "alpha": 32,
        "dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    },

    # Training parameters
    "training": {
        "per_device_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "max_length": 2048,
        "warmup_steps": 1000,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": 10,
        "eval_steps": 500,
        "save_steps": 500,
        "max_hours": 12
    },

    # ChatML system prompt
    "system_prompt": "You are Wendy, an empathetic and supportive mental health companion. You provide thoughtful, therapeutic responses while maintaining appropriate boundaries."
}

# Global state
shutdown_requested = False
training_start_time = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_requested = True


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ===========================================
# Dataset Loading (from wayfarer_supervised.py)
# ===========================================

def find_dataset_files(base_dir: str, patterns: List[str]) -> List[str]:
    """Find dataset files matching patterns."""
    files = []
    for pattern in patterns:
        full_pattern = os.path.join(base_dir, pattern)
        matches = glob.glob(full_pattern, recursive=True)
        files.extend(matches)

        # Also check without pattern if it's a direct path
        direct_path = os.path.join(base_dir, pattern)
        if os.path.isfile(direct_path) and direct_path not in files:
            files.append(direct_path)

    return files


def load_dataset_file(file_path: str) -> List[Dict[str, Any]]:
    """Load a single dataset file (JSON, JSONL, or CSV)."""
    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".jsonl":
            with open(file_path, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif ext == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Handle both list format and dict with conversations key
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    if "conversations" in data:
                        return data["conversations"]
                    return [data]
        elif ext == ".csv":
            df = pd.read_csv(file_path)
            return [{str(k): v for k, v in rec.items()} for rec in df.to_dict(orient="records")]
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")

    return []


def clean_and_normalize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean and normalize records, removing PII."""
    cleaned = []
    pii_fields = {"email", "phone", "ssn", "name", "user_id", "ip_address"}

    for rec in records:
        # Normalize field names
        rec = {k.lower().strip(): v for k, v in rec.items()}

        # Remove PII fields
        for field in pii_fields:
            rec.pop(field, None)

        # Normalize text
        for k in rec:
            if isinstance(rec[k], str):
                rec[k] = rec[k].replace("\r\n", "\n").strip()

        cleaned.append(rec)

    return cleaned


def deduplicate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate records."""
    seen = set()
    deduped = []

    for rec in records:
        # Create a hashable key from the record
        key = json.dumps(rec, sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(rec)

    return deduped


def convert_to_chatml(records: List[Dict[str, Any]], system_prompt: str) -> List[Dict[str, Any]]:
    """Convert records to ChatML format."""
    chatml_data = []

    for rec in records:
        messages = [{"role": "system", "content": system_prompt}]

        # Handle different input formats
        if "conversation" in rec:
            for turn in rec["conversation"]:
                role = turn.get("role", "user")
                # Normalize role names
                if role in ["client", "user", "human"]:
                    role = "user"
                elif role in ["therapist", "assistant", "ai", "wendy"]:
                    role = "assistant"
                messages.append({"role": role, "content": turn.get("content", "")})

        elif "messages" in rec:
            # Already in message format
            messages.extend(rec["messages"])

        elif "input" in rec and "output" in rec:
            messages.append({"role": "user", "content": rec["input"]})
            messages.append({"role": "assistant", "content": rec["output"]})

        elif "question" in rec and "answer" in rec:
            messages.append({"role": "user", "content": rec["question"]})
            messages.append({"role": "assistant", "content": rec["answer"]})

        elif "context" in rec:
            # CoT format with reasoning
            content = rec.get("context", "")
            if "reasoning" in rec:
                content += f"\n\n[Reasoning: {rec['reasoning']}]"
            if "response" in rec:
                messages.append({"role": "user", "content": content})
                messages.append({"role": "assistant", "content": rec["response"]})

        else:
            # Try to extract any text field
            for field in ["text", "content", "dialogue"]:
                if field in rec and rec[field]:
                    messages.append({"role": "user", "content": str(rec[field])})
                    break

        if len(messages) > 1:  # More than just system prompt
            chatml_data.append({"messages": messages})

    return chatml_data


def load_stage_datasets(data_dir: str, dataset_patterns: List[str], system_prompt: str) -> List[Dict[str, Any]]:
    """Load and process datasets for a training stage."""
    logger.info(f"Loading datasets: {dataset_patterns}")

    all_records = []
    files = find_dataset_files(data_dir, dataset_patterns)

    if not files:
        logger.warning(f"No dataset files found for patterns: {dataset_patterns}")
        return []

    for file_path in files:
        records = load_dataset_file(file_path)
        logger.info(f"Loaded {len(records)} records from {file_path}")
        all_records.extend(records)

    # Clean and deduplicate
    cleaned = clean_and_normalize(all_records)
    deduped = deduplicate(cleaned)
    logger.info(f"After cleaning/dedup: {len(deduped)} records")

    # Convert to ChatML
    chatml = convert_to_chatml(deduped, system_prompt)
    logger.info(f"Converted to ChatML: {len(chatml)} conversations")

    return chatml


# ===========================================
# Training Logic
# ===========================================

def setup_wandb(config: Dict, stage_name: str):
    """Initialize Weights & Biases logging."""
    try:
        import wandb

        wandb_key = os.environ.get("WANDB_API_KEY")
        if not wandb_key:
            logger.warning("WANDB_API_KEY not set. Skipping WandB initialization.")
            return None

        run = wandb.init(
            project="pixelated-empathy",
            name=f"ovh-{stage_name}-{datetime.now().strftime('%Y%m%d-%H%M')}",
            config=config,
            tags=["ovh", stage_name, "wayfarer-2-12b"]
        )
        logger.info(f"WandB initialized: {run.url}")
        return run
    except Exception as e:
        logger.warning(f"Failed to initialize WandB: {e}")
        return None


def create_model(config: Dict):
    """Create or load the model with LoRA configuration."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        base_model = config.get("base_model", DEFAULT_CONFIG["base_model"])
        lora_config = config.get("lora", DEFAULT_CONFIG["lora"])

        logger.info(f"Loading base model: {base_model}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with appropriate settings for GPU
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare for training
        model = prepare_model_for_kbit_training(model)

        # Apply LoRA
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            lora_dropout=lora_config["dropout"],
            target_modules=lora_config["target_modules"],
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, peft_config)

        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return model, tokenizer

    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        logger.error("Install with: pip install transformers peft accelerate bitsandbytes")
        raise


def train_stage(
    stage_name: str,
    model,
    tokenizer,
    train_data: List[Dict],
    config: Dict,
    checkpoint_dir: str,
    resume_from: Optional[str] = None
):
    """Train a single stage."""
    from transformers import Trainer, TrainingArguments
    from datasets import Dataset

    global shutdown_requested

    stage_config = config.get("training_stages", {}).get(stage_name, {})
    training_config = config.get("training", DEFAULT_CONFIG["training"])

    epochs = stage_config.get("epochs", 3)
    lr = stage_config.get("learning_rate", 2e-4)

    logger.info(f"=" * 60)
    logger.info(f"Training Stage: {stage_name}")
    logger.info(f"Description: {stage_config.get('description', 'N/A')}")
    logger.info(f"Epochs: {epochs}, Learning Rate: {lr}")
    logger.info(f"Samples: {len(train_data)}")
    logger.info(f"=" * 60)

    # Create output directory
    output_dir = os.path.join(checkpoint_dir, stage_name)
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataset
    def tokenize_function(examples):
        # Flatten messages to text
        texts = []
        for messages in examples["messages"]:
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"<|{role}|>\n{content}\n<|end|>\n"
            texts.append(text)

        result = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=training_config.get("max_length", 2048)
        )
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = Dataset.from_list(train_data)
    tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["messages"])

    # Split train/eval
    split = tokenized.train_test_split(test_size=0.1, seed=42)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=training_config.get("per_device_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 8),
        learning_rate=lr,
        warmup_steps=training_config.get("warmup_steps", 1000),
        weight_decay=training_config.get("weight_decay", 0.01),
        max_grad_norm=training_config.get("max_grad_norm", 1.0),
        logging_steps=training_config.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=training_config.get("eval_steps", 500),
        save_strategy="steps",
        save_steps=training_config.get("save_steps", 500),
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"{stage_name}-{datetime.now().strftime('%Y%m%d')}",
        resume_from_checkpoint=resume_from
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        tokenizer=tokenizer
    )

    # Train
    if not shutdown_requested:
        trainer.train(resume_from_checkpoint=resume_from)

        # Save final checkpoint
        final_path = os.path.join(output_dir, "final")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logger.info(f"Saved stage checkpoint to: {final_path}")

        return final_path

    return None


# ===========================================
# Main Entry Point
# ===========================================

def main():
    global training_start_time

    parser = argparse.ArgumentParser(description="OVH AI Training - Wayfarer-2-12B SFT")
    parser.add_argument("--config", type=str, help="Path to config JSON file")
    parser.add_argument("--stage", type=str, choices=["foundation", "reasoning", "voice", "all"],
                        default="all", help="Training stage to run")
    parser.add_argument("--data-dir", type=str, default="/data", help="Data directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/checkpoints", help="Checkpoint directory")
    parser.add_argument("--resume-from", type=str, help="Resume from checkpoint path")
    args = parser.parse_args()

    training_start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("OVH AI Training - Wayfarer-2-12B Supervised Fine-Tuning")
    logger.info(f"Start time: {training_start_time}")
    logger.info("=" * 60)

    # Load config
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()

    # Override paths from args
    config["data_dir"] = args.data_dir
    config["checkpoint_dir"] = args.checkpoint_dir

    # Check CUDA
    if not torch.cuda.is_available():
        logger.error("CUDA not available. This script requires GPU.")
        sys.exit(1)

    logger.info(f"CUDA available: {torch.cuda.device_count()} device(s)")
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")

    # Initialize WandB
    wandb_run = setup_wandb(config, args.stage)

    try:
        # Create model
        model, tokenizer = create_model(config)

        # Determine stages to run
        stages_to_run = []
        if args.stage == "all":
            stages_to_run = ["foundation", "reasoning", "voice"]
        else:
            stages_to_run = [args.stage]

        # Run training stages
        last_checkpoint = args.resume_from

        for stage_name in stages_to_run:
            if shutdown_requested:
                logger.warning("Shutdown requested, stopping training")
                break

            stage_config = config.get("training_stages", {}).get(stage_name, {})
            datasets = stage_config.get("datasets", [])

            # Load datasets
            train_data = load_stage_datasets(
                args.data_dir,
                datasets,
                config.get("system_prompt", DEFAULT_CONFIG["system_prompt"])
            )

            if not train_data:
                logger.warning(f"No data found for stage '{stage_name}', skipping")
                continue

            # Train stage
            checkpoint = train_stage(
                stage_name=stage_name,
                model=model,
                tokenizer=tokenizer,
                train_data=train_data,
                config=config,
                checkpoint_dir=args.checkpoint_dir,
                resume_from=last_checkpoint
            )

            if checkpoint:
                last_checkpoint = checkpoint

        # Final save
        if last_checkpoint:
            final_model_path = os.path.join(args.checkpoint_dir, "final_model")
            logger.info(f"Saving final model to: {final_model_path}")
            model.save_pretrained(final_model_path)
            tokenizer.save_pretrained(final_model_path)

        logger.info("=" * 60)
        logger.info("Training completed successfully!")
        logger.info(f"Total duration: {datetime.now() - training_start_time}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    finally:
        if wandb_run:
            import wandb
            wandb.finish()


if __name__ == "__main__":
    main()
