#!/usr/bin/env python3
"""
Lightning.ai H100 Therapeutic AI Training Script
4-Expert MoE LoRA training for therapeutic conversation AI
"""

import json
import torch
import lightning as L
from lightning.fabric import Fabric
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
import logging
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TherapeuticConversationDataset(Dataset):
    """Dataset for therapeutic conversation training"""
    
    def __init__(self, conversations: List[Dict], tokenizer, max_length: int = 1024):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Format conversation for training
        if 'conversations' in conversation:
            # Standard format
            text_parts = []
            for turn in conversation['conversations']:
                role = "Human" if turn['from'] == 'human' else "Assistant"
                text_parts.append(f"{role}: {turn['value']}")
            full_text = "\n".join(text_parts)
        else:
            # Fallback format
            full_text = conversation.get('text', str(conversation))
        
        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),
            'expert_id': conversation.get('expert_id', 0),
            'quality_score': conversation.get('computed_quality', 0.5)
        }

class TherapeuticTrainer(L.LightningModule):
    """Lightning trainer for therapeutic AI with MoE LoRA"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Initialize model and tokenizer
        model_name = config['model_config']['base_model']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config['model_config']['lora_r'],
            lora_alpha=config['model_config']['lora_alpha'],
            lora_dropout=config['model_config']['lora_dropout'],
            target_modules=config['model_config']['target_modules']
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        logger.info(f"✅ Model initialized: {model_name} with LoRA")
        logger.info(f"   Trainable parameters: {self.model.num_parameters()}")
    
    def forward(self, batch):
        return self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_perplexity', torch.exp(loss), prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = outputs.loss
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_perplexity', torch.exp(loss), prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config['training_config']['learning_rate'],
            weight_decay=self.config['training_config']['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config['training_config']['num_epochs']
        )
        
        return [optimizer], [scheduler]

def load_datasets(data_dir: Path) -> Dict[str, List[Dict]]:
    """Load training and validation datasets"""
    datasets = {}
    
    # Load main datasets
    train_path = data_dir / "train.json"
    val_path = data_dir / "validation.json"
    
    for name, path in [("train", train_path), ("validation", val_path)]:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                datasets[name] = json.load(f)
            logger.info(f"✅ Loaded {name}: {len(datasets[name])} conversations")
        else:
            logger.error(f"❌ Missing {name} dataset: {path}")
            raise FileNotFoundError(f"Required dataset not found: {path}")
    
    return datasets

def main():
    """Main training function"""
    logger.info("🚀 Starting Lightning.ai H100 Therapeutic AI Training")
    
    # Load configuration
    config_path = Path("unified_lightning_config.json")
    if not config_path.exists():
        raise FileNotFoundError("Configuration file not found")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load datasets
    datasets = load_datasets(Path("."))
    
    # Initialize tokenizer
    model_name = config['model_config']['base_model']
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    train_dataset = TherapeuticConversationDataset(
        datasets['train'], 
        tokenizer, 
        config['training_config']['max_length']
    )
    val_dataset = TherapeuticConversationDataset(
        datasets['validation'], 
        tokenizer, 
        config['training_config']['max_length']
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training_config']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training_config']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Initialize model
    model = TherapeuticTrainer(config)
    
    # Configure trainer
    trainer = L.Trainer(
        max_epochs=config['training_config']['num_epochs'],
        accelerator="gpu",
        devices=1,  # H100
        precision=16,
        gradient_clip_val=1.0,
        accumulate_grad_batches=config['training_config']['gradient_accumulation_steps'],
        val_check_interval=config['training_config']['eval_steps'],
        log_every_n_steps=config['training_config']['logging_steps'],
        enable_checkpointing=True,
        default_root_dir="./lightning_logs"
    )
    
    # Start training
    logger.info("🔥 Starting H100 training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Save final model
    model.model.save_pretrained("./therapeutic_ai_final")
    tokenizer.save_pretrained("./therapeutic_ai_final")
    
    logger.info("🎉 Training complete! Model saved to ./therapeutic_ai_final")

if __name__ == "__main__":
    main()
