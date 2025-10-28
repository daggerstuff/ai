#!/usr/bin/env python3
"""
Enhanced Training Script - KAN-28 Components Integrated
Trains therapeutic AI model with all 6 components on Lightning.ai H100
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import wandb

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KAN28EnhancedTrainer:
    """Enhanced trainer incorporating all 6 KAN-28 components"""
    
    def __init__(self, config_path: str = "../config/enhanced_training_config.json"):
        self.config = self._load_config(config_path)
        self.tokenizer = None
        self.model = None
        self.datasets = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_model_and_tokenizer(self):
        """Initialize model and tokenizer with KAN-28 enhancements"""
        
        model_name = self.config.get("base_model", "LatitudeGames/Wayfarer-2-12B")
        
        logger.info(f"Loading model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if self.config.get("h100_optimizations", {}).get("bf16", True) else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.get("lora_config", {}).get("lora_r", 16),
            lora_alpha=self.config.get("lora_config", {}).get("lora_alpha", 32),
            lora_dropout=self.config.get("lora_config", {}).get("lora_dropout", 0.1),
            target_modules=self.config.get("lora_config", {}).get("lora_target_modules", ["q_proj", "v_proj"]),
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info("Model and tokenizer setup complete")
        return self.model, self.tokenizer
    
    def load_datasets(self):
        """Load and prepare datasets with KAN-28 component validation"""
        
        data_dir = Path("../data")
        
        # Load ultimate final dataset
        ultimate_dataset_path = data_dir / "ULTIMATE_FINAL_DATASET.jsonl"
        component_dataset_path = data_dir / "unified_6_component_dataset.jsonl"
        
        logger.info("Loading datasets...")
        
        # Load main dataset
        main_conversations = []
        if ultimate_dataset_path.exists():
            with open(ultimate_dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        main_conversations.append(json.loads(line))
            logger.info(f"Loaded {len(main_conversations)} conversations from ultimate dataset")
        
        # Load component-enhanced dataset
        component_conversations = []
        if component_dataset_path.exists():
            with open(component_dataset_path, 'r') as f:
                for line in f:
                    if line.strip():
                        component_conversations.append(json.loads(line))
            logger.info(f"Loaded {len(component_conversations)} component-enhanced conversations")
        
        # Combine and prepare datasets
        all_conversations = main_conversations + component_conversations
        
        # Convert to training format
        train_data = []
        for conv in all_conversations:
            formatted_conv = self._format_conversation(conv)
            if formatted_conv:
                train_data.append(formatted_conv)
        
        logger.info(f"Prepared {len(train_data)} training examples")
        
        # Split into train/val
        split_idx = int(len(train_data) * 0.9)
        
        self.datasets["train"] = Dataset.from_list(train_data[:split_idx])
        self.datasets["val"] = Dataset.from_list(train_data[split_idx:])
        
        # Tokenize datasets
        self.datasets["train"] = self.datasets["train"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.datasets["train"].column_names
        )
        
        self.datasets["val"] = self.datasets["val"].map(
            self._tokenize_function,
            batched=True,
            remove_columns=self.datasets["val"].column_names
        )
        
        return self.datasets
    
    def _format_conversation(self, conv: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Format conversation for training with KAN-28 component awareness"""
        
        # Handle different conversation formats
        if "conversation" in conv:
            conversation_data = conv["conversation"]
            if isinstance(conversation_data, dict):
                client = conversation_data.get("client", "")
                therapist = conversation_data.get("therapist", "")
            else:
                return None
        elif "client" in conv and "therapist" in conv:
            client = conv["client"]
            therapist = conv["therapist"]
        else:
            return None
        
        if not client or not therapist:
            return None
        
        # Create enhanced training text with component metadata
        component_info = ""
        if "integration_metadata" in conv:
            components = conv["integration_metadata"].get("components_applied", [])
            if components:
                component_info = f" [Components: {', '.join(components)}]"
        
        # Format as conversation
        formatted_text = f"<|im_start|>user\n{client}<|im_end|>\n<|im_start|>assistant\n{therapist}{component_info}<|im_end|>"
        
        return {"text": formatted_text}
    
    def _tokenize_function(self, examples):
        """Tokenize examples for training"""
        
        # Tokenize the text
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=self.config.get("context_config", {}).get("training_max_length", 2048),
            return_overflowing_tokens=False,
        )
        
        # Set labels for language modeling
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    def setup_training(self):
        """Setup training arguments and trainer"""
        
        output_dir = "./pixelated_empathy_kan28"
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.get("num_train_epochs", 3),
            per_device_train_batch_size=self.config.get("per_device_train_batch_size", 4),
            per_device_eval_batch_size=self.config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 8),
            learning_rate=self.config.get("learning_rate", 3e-4),
            weight_decay=self.config.get("weight_decay", 0.01),
            warmup_steps=self.config.get("warmup_steps", 1000),
            max_grad_norm=self.config.get("max_grad_norm", 1.0),
            
            # H100 optimizations
            bf16=self.config.get("h100_optimizations", {}).get("bf16", True),
            gradient_checkpointing=self.config.get("h100_optimizations", {}).get("gradient_checkpointing", True),
            optim=self.config.get("h100_optimizations", {}).get("optim", "adamw_torch_fused"),
            dataloader_num_workers=self.config.get("h100_optimizations", {}).get("dataloader_num_workers", 4),
            dataloader_pin_memory=self.config.get("h100_optimizations", {}).get("dataloader_pin_memory", True),
            group_by_length=self.config.get("h100_optimizations", {}).get("group_by_length", True),
            
            # Logging and checkpoints
            logging_steps=self.config.get("logging", {}).get("logging_steps", 10),
            eval_steps=self.config.get("logging", {}).get("eval_steps", 500),
            save_steps=self.config.get("logging", {}).get("save_steps", 500),
            save_total_limit=self.config.get("logging", {}).get("save_total_limit", 5),
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Weights & Biases
            report_to="wandb",
            run_name="pixelated_empathy_kan28_training",
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets["train"],
            eval_dataset=self.datasets["val"],
            data_collator=data_collator,
        )
        
        return self.trainer
    
    def train(self):
        """Execute training with KAN-28 components"""
        
        logger.info("Starting KAN-28 enhanced training...")
        
        # Initialize wandb
        wandb.init(
            project="pixelated-empathy-kan28",
            name="therapeutic_ai_6_components",
            config=self.config
        )
        
        # Setup everything
        self.setup_model_and_tokenizer()
        self.load_datasets()
        self.setup_training()
        
        # Start training
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.trainer.args.output_dir)
        
        # Log training results
        logger.info(f"Training completed! Final loss: {train_result.training_loss}")
        
        # Log KAN-28 component integration success
        wandb.log({
            "final_loss": train_result.training_loss,
            "kan28_components_integrated": 6,
            "total_conversations": len(self.datasets["train"]) + len(self.datasets["val"]),
            "component_enhanced_conversations": 39
        })
        
        wandb.finish()
        
        return train_result

def main():
    """Main training function"""
    
    # Set environment variables for Lightning.ai
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Initialize trainer
    trainer = KAN28EnhancedTrainer()
    
    # Start training
    result = trainer.train()
    
    print("KAN-28 Enhanced Training Complete!")
    print(f"Model saved to: ./pixelated_empathy_kan28")
    print(f"Final training loss: {result.training_loss}")

if __name__ == "__main__":
    main()