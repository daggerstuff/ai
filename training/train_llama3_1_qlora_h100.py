#!/usr/bin/env python3
"""
Llama 3.1 8B QLoRA Training Script for H100
Optimized for 80GB VRAM and high throughput
"""

import json
import logging
import signal
import time
from datetime import UTC, datetime

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("llama3_1_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("\n🛑 Shutdown requested")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

class TimeConstraintCallback(TrainerCallback):
    """Callback to enforce training window and handle graceful shutdown"""
    def __init__(self, max_hours: int = 12):
        self.max_hours = max_hours
        self.start_time = None
        self.last_checkpoint_time = None
        self.checkpoint_interval_minutes = 30

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        logger.info(f"⏰ Training started at {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏰ Maximum training duration: {self.max_hours} hours")

    def on_step_end(self, args, state, control, **kwargs):
        global shutdown_requested
        if shutdown_requested:
            logger.info("🛑 Stopping training due to shutdown request")
            control.should_training_stop = True
            control.should_save = True
            return control

        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600

        if elapsed_hours >= self.max_hours - 0.5:
            logger.info(f"⏰ Approaching {self.max_hours}-hour limit. Stopping training...")
            control.should_training_stop = True
            control.should_save = True

        elapsed_since_checkpoint = (current_time - self.last_checkpoint_time) / 60
        if elapsed_since_checkpoint >= self.checkpoint_interval_minutes:
            control.should_save = True
            self.last_checkpoint_time = current_time
            logger.info(f"💾 Checkpoint at {elapsed_hours:.2f} hours")

        return control

def load_training_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)

def load_tokenizer(model_name: str) -> AutoTokenizer:
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def preprocess_function(examples, tokenizer, max_seq_length: int):
    inputs = []
    for i in range(len(examples["text"] if "text" in examples else examples["conversations"])):
        if "text" in examples:
            inputs.append(examples["text"][i])
        else:
            # Handle chat format
            conv = examples["conversations"][i]
            conversation_text = ""
            for turn in conv:
                conversation_text += f"{turn['role']}: {turn['content']}\n"
            inputs.append(conversation_text.strip())

    return tokenizer(
        inputs,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )

def main(config_path: str):
    try:
        config = load_training_config(config_path)
        logger.info("Config loaded successfully")

        # Device check
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. H100 required.")

        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

        # Tokenizer
        tokenizer = load_tokenizer(config["model"]["base_model"])

        # Datasets
        logger.info("Loading datasets...")
        dataset_args = {}
        if config["data"]["train_file"].endswith(".jsonl"):
            dataset_args["data_files"] = {"train": config["data"]["train_file"], "validation": config["data"]["validation_file"]}
            dataset = load_dataset("json", **dataset_args)
        else:
            # Assume local folder or other format
            dataset = load_dataset(config["data"]["train_file"])

        logger.info("Preprocessing datasets...")
        tokenized_dataset = dataset.map(
            lambda x: preprocess_function(examples=x, tokenizer=tokenizer, max_seq_length=config["data"]["max_seq_length"]),
            batched=True,
            remove_columns=dataset["train"].column_names,
            num_proc=config["data"]["preprocessing_num_workers"]
        )

        # Quantization Config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=config["model"]["load_in_4bit"],
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config["model"]["torch_dtype"] == "bfloat16" else torch.float16
        )

        # Load Model
        logger.info(f"Loading base model: {config['model']['base_model']}")
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if config["model"]["torch_dtype"] == "bfloat16" else torch.float16,
            trust_remote_code=True,
            use_flash_attention_2=torch.cuda.get_device_capability()[0] >= 8
        )

        # Prepare for k-bit training
        model = prepare_model_for_kbit_training(model)

        # LoRA Config
        lora_config = LoraConfig(
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
            lora_dropout=config["lora"]["lora_dropout"],
            bias=config["lora"]["bias"],
            task_type=config["lora"]["task_type"]
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # Training Arguments
        training_args = TrainingArguments(
            output_dir=config["training"]["output_dir"],
            num_train_epochs=config["training"]["num_train_epochs"],
            per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
            per_device_eval_batch_size=config["training"]["per_device_eval_batch_size"],
            gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
            learning_rate=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
            warmup_ratio=config["training"]["warmup_ratio"],
            lr_scheduler_type=config["training"]["lr_scheduler_type"],
            logging_steps=config["training"]["logging_steps"],
            evaluation_strategy=config["training"]["evaluation_strategy"],
            eval_steps=config["training"]["eval_steps"],
            save_steps=config["training"]["save_steps"],
            save_total_limit=config["training"]["save_total_limit"],
            load_best_model_at_end=config["training"]["load_best_model_at_end"],
            metric_for_best_model=config["training"]["metric_for_best_model"],
            report_to=config["training"]["report_to"],
            run_name=config["training"]["run_name"],
            bf16=config["system"]["bf16"],
            gradient_checkpointing=config["system"]["gradient_checkpointing"],
            optim=config["system"]["optim"],
            group_by_length=config["system"]["group_by_length"],
            dataloader_num_workers=config["system"]["dataloader_num_workers"],
            dataloader_pin_memory=config["system"]["dataloader_pin_memory"],
            ddp_find_unused_parameters=False,
        )

        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
            callbacks=[TimeConstraintCallback(max_hours=12)]
        )

        # Start Training
        logger.info("Starting training...")
        trainer.train()

        # Save Final Model
        logger.info(f"Saving final model to {config['training']['output_dir']}")
        trainer.save_model()
        tokenizer.save_pretrained(config["training"]["output_dir"])

        logger.info("✅ Training completed successfully")

    except Exception as e:
        logger.error(f"❌ Training failed: {e!s}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Llama 3.1 8B QLoRA Training Script")
    parser.add_argument("--config", type=str, default="ai/configs/llama3_1_qlora_h100.json", help="Path to config file")
    args = parser.parse_args()
    main(args.config)
