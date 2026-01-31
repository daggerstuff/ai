#!/usr/bin/env python3
"""
Training Script for Pixelated Empathy AI on Lightning.ai H100

This script prepares and executes training on H100 GPUs using PyTorch Lightning
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset, load_dataset
import wandb

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmpathyAITrainer:
    """Trainer for Pixelated Empathy AI therapeutic conversation model"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path) if config_path else self._default_config()
        self.tokenizer = None
        self.model = None
        self.trainer = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration from YAML/JSON file"""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                import yaml
                return yaml.safe_load(f)
            else:
                return json.load(f)

    def _default_config(self) -> Dict[str, Any]:
        """Default training configuration"""
        return {
            "model": {
                "base_model": "mistralai/Mixtral-8x22B-Instruct-v0.1",
                "tokenizer": "mistralai/Mixtral-8x22B-Instruct-v0.1"
            },
            "training": {
                "batch_size": 1024,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 64,
                "learning_rate": 1e-5,
                "num_epochs": 3,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "max_seq_length": 4096,
                "gradient_checkpointing": True,
                "precision": "bf16"
            },
            "datasets": {
                "training": {
                    "path": "ai/training_data_consolidated/final_datasets/ULTIMATE_FINAL_DATASET.jsonl"
                },
                "validation": {
                    "path": "ai/training_data_consolidated/final_datasets/pixelated_empathy_val_20250526_174637.jsonl"
                }
            },
            "output": {
                "model_artifacts": {
                    "path": "./model_output"
                },
                "logs": {
                    "path": "./logs"
                }
            }
        }

    def initialize_model_and_tokenizer(self):
        """Initialize the model and tokenizer"""
        logger.info(f"Loading model: {self.config['model']['base_model']}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model']['tokenizer'],
            trust_remote_code=True
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model']['base_model'],
            torch_dtype=torch.bfloat16 if self.config['training']['precision'] == 'bf16' else torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        logger.info("Model and tokenizer loaded successfully")

    def prepare_datasets(self):
        """Prepare training and validation datasets"""
        logger.info("Preparing datasets...")

        # Load training dataset
        train_dataset_path = self.config['datasets']['training']['path']
        logger.info(f"Loading training dataset: {train_dataset_path}")

        # Load dataset
        train_dataset = load_dataset('json', data_files=train_dataset_path, split='train')

        # Process dataset
        def preprocess_function(examples):
            # Format the data appropriately
            texts = []
            for i in range(len(examples['category'])):
                # Format based on the actual data structure
                if 'instructions' in examples:
                    text = examples['instructions'][i]
                elif 'text' in examples:
                    text = examples['text'][i]
                else:
                    text = str(examples)[i] if isinstance(examples, list) else str(examples)

                texts.append(text)

            # Tokenize
            return self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=self.config['training']['max_seq_length'],
                return_tensors="pt"
            )

        # Apply preprocessing
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )

        # Load validation dataset if available
        val_dataset = None
        if 'validation' in self.config['datasets']:
            val_dataset_path = self.config['datasets']['validation']['path']
            logger.info(f"Loading validation dataset: {val_dataset_path}")
            val_dataset = load_dataset('json', data_files=val_dataset_path, split='train')
            val_dataset = val_dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=val_dataset.column_names
            )

        logger.info(f"Datasets prepared - Train: {len(train_dataset)}, Val: {len(val_dataset) if val_dataset else 0}")
        return train_dataset, val_dataset

    def setup_trainer(self, train_dataset, val_dataset=None):
        """Setup the trainer with training arguments"""
        logger.info("Setting up trainer...")

        # Create output directories
        output_dir = Path(self.config['output']['model_artifacts']['path'])
        output_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = Path(self.config['output']['logs']['path'])
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['training']['per_device_train_batch_size'],
            per_device_eval_batch_size=self.config['training']['per_device_train_batch_size'],
            gradient_accumulation_steps=self.config['training']['gradient_accumulation_steps'],
            warmup_steps=self.config['training']['warmup_steps'],
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=self.config['training']['weight_decay'],
            logging_dir=str(logs_dir),
            logging_steps=10,
            save_steps=500,
            save_total_limit=5,
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=1000 if val_dataset else None,
            save_strategy="steps",
            load_best_model_at_end=True if val_dataset else False,
            dataloader_num_workers=4,
            fp16=self.config['training']['precision'] == 'fp16',
            bf16=self.config['training']['precision'] == 'bf16',
            gradient_checkpointing=self.config['training']['gradient_checkpointing'],
            report_to=["wandb"] if self._wandb_available() else ["tensorboard"],
            dataloader_pin_memory=True,
            remove_unused_columns=False,
            push_to_hub=False,
            hub_model_id=None,
            hub_strategy="every_save",
            hub_token=None,
            hub_private_repo=False,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        logger.info("Trainer setup completed")

    def _wandb_available(self) -> bool:
        """Check if Weights & Biases is available"""
        try:
            import wandb
            return True
        except ImportError:
            return False

    def train(self):
        """Execute the training process"""
        logger.info("Starting training process...")

        try:
            # Initialize Weights & Biases if available
            if self._wandb_available():
                wandb.init(
                    project="pixelated-empathy-ai",
                    name="h100-training-run",
                    config=self.config
                )

            # Start training
            train_result = self.trainer.train()

            # Save model
            self.trainer.save_model()
            self.trainer.state.save_to_json(os.path.join(self.config['output']['model_artifacts']['path'], "trainer_state.json"))

            logger.info("Training completed successfully")
            logger.info(f"Training metrics: {train_result.metrics}")

            return train_result

        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

    def evaluate(self):
        """Evaluate the trained model"""
        if self.trainer is None:
            logger.warning("Trainer not initialized. Cannot evaluate.")
            return None

        logger.info("Evaluating model...")
        eval_result = self.trainer.evaluate()
        logger.info(f"Evaluation results: {eval_result}")
        return eval_result

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Train Pixelated Empathy AI on H100")
    parser.add_argument("--config", type=str, help="Path to training configuration file")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    args = parser.parse_args()

    print("🚀 Pixelated Empathy AI Training - Lightning.ai H100 Deployment")
    print("=" * 60)

    try:
        # Initialize trainer
        trainer = EmpathyAITrainer(args.config)

        # Initialize model and tokenizer
        trainer.initialize_model_and_tokenizer()

        # Prepare datasets
        train_dataset, val_dataset = trainer.prepare_datasets()

        # Setup trainer
        trainer.setup_trainer(train_dataset, val_dataset)

        # Execute training
        train_result = trainer.train()

        # Evaluate if requested
        if args.evaluate:
            trainer.evaluate()

        print("\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {trainer.config['output']['model_artifacts']['path']}")
        print(f"📊 Training loss: {train_result.training_loss:.4f}")
        print(f"⏰ Total training time: {train_result.metrics.get('train_runtime', 'N/A')} seconds")

        return True

    except Exception as e:
        logger.error(f"Training process failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)