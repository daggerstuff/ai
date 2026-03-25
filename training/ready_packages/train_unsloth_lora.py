#!/usr/bin/env python3
"""
Unsloth LoRA/QLoRA Training Script
Train Pixelated Empathy AI model using Unsloth with LoRA/QLoRA fine-tuning
"""

import os
import logging
import json
from typing import Optional
from pathlib import Path

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("unsloth_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_training_config(config_path: str = "training_config.json") -> dict:
    """Load training configuration from JSON file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def load_tokenizer(model_name: str) -> AutoTokenizer:
    """Load and configure tokenizer"""
    logger.info(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def load_datasets(config: dict) -> tuple[Dataset, Dataset]:
    """Load training and validation datasets"""
    logger.info("Loading training datasets")
    
    train_files = list(Path(config["data"]["train_file"].split("*")[0]).parent.glob(
        config["data"]["train_file"].split("*")[-1]
    ))
    val_files = list(Path(config["data"]["validation_file"].split("*")[0]).parent.glob(
        config["data"]["validation_file"].split("*")[-1]
    ))
    
    logger.info(f"Found {len(train_files)} training files")
    logger.info(f"Found {len(val_files)} validation files")
    
    train_dataset = load_dataset("json", data_files=[str(f) for f in train_files], split="train")
    val_dataset = load_dataset("json", data_files=[str(f) for f in val_files], split="train")
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    return train_dataset, val_dataset

def preprocess_function(examples, tokenizer, max_seq_length: int):
    """Preprocess function for tokenizing examples"""
    # Combine conversation history and user query
    inputs = []
    for example in examples:
        if "text" in example:
            inputs.append(example["text"])
        elif "conversations" in example:
            # Handle chat format
            conversation_text = ""
            for turn in example["conversations"]:
                conversation_text += f"{turn['role']}: {turn['content']}\n"
            inputs.append(conversation_text.strip())
        else:
            logger.warning(f"Example missing text or conversations field: {example}")
            inputs.append("")
    
    tokenized = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=max_seq_length,
        return_tensors="pt"
    )
    
    # Set labels
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def create_bnb_config() -> BitsAndBytesConfig:
    """Create BitsAndBytes configuration for 4-bit quantization"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

def create_lora_config(config: dict) -> LoraConfig:
    """Create LoRA configuration"""
    return LoraConfig(
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        task_type=config["lora"]["task_type"]
    )

def create_training_args(config: dict) -> TrainingArguments:
    """Create training arguments"""
    return TrainingArguments(
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
        greater_is_better=config["training"]["greater_is_better"],
        report_to=config["training"]["report_to"],
        run_name=config["training"]["run_name"],
        fp16=config["system"]["fp16"],
        bf16=config["system"]["bf16"],
        gradient_checkpointing=config["system"]["gradient_checkpointing"],
        remove_unused_columns=config["system"]["remove_unused_columns"],
        dataloader_num_workers=config["system"]["dataloader_num_workers"],
        dataloader_pin_memory=config["system"]["dataloader_pin_memory"],
    )

def main(config_path: str = "training_config.json"):
    """Main training function"""
    try:
        # Load training configuration
        config = load_training_config(config_path)
        logger.info("Training configuration loaded successfully")
        
        # Load tokenizer
        tokenizer = load_tokenizer(config["model"]["base_model"])
        
        # Load datasets
        train_dataset, val_dataset = load_datasets(config)
        
        # Preprocess datasets
        logger.info("Preprocessing training dataset")
        tokenized_train = train_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, config["data"]["max_seq_length"]),
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=config["data"]["preprocessing_num_workers"]
        )
        
        logger.info("Preprocessing validation dataset")
        tokenized_val = val_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer, config["data"]["max_seq_length"]),
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=config["data"]["preprocessing_num_workers"]
        )
        
        # Load model with quantization
        logger.info(f"Loading model: {config['model']['base_model']}")
        bnb_config = create_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(
            config["model"]["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA
        logger.info("Applying LoRA configuration")
        lora_config = create_lora_config(config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Create training arguments and trainer
        training_args = create_training_args(config)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            tokenizer=tokenizer,
        )
        
        # Start training
        logger.info("Starting training")
        trainer.train()
        
        # Save final model
        logger.info("Training completed, saving final model")
        final_model_path = os.path.join(config["training"]["output_dir"], "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        
        logger.info("Training script completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unsloth LoRA/QLoRA Training Script")
    parser.add_argument("--config", type=str, default="training_config.json", help="Path to training configuration file")
    args = parser.parse_args()
    
    main(args.config)
