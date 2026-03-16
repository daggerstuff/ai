#!/usr/bin/env python3
"""
Clean, deduplicate, and format final training dataset
"""

import json
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Set

class DatasetCleaner:
    def __init__(self):
        self.seen_hashes: Set[str] = set()
        self.duplicates_removed = 0
        self.cleaned_count = 0
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for consistent formatting"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Fix common transcription artifacts
        text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[,\s]+', ' ', text)  # Multiple commas/spaces
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace again
        
        # Remove very short or empty responses
        if len(text.strip()) < 50:
            return ""
            
        return text.strip()
    
    def get_content_hash(self, conversation: Dict) -> str:
        """Generate hash for duplicate detection"""
        human_text = conversation["conversations"][0]["value"]
        gpt_text = conversation["conversations"][1]["value"]
        
        # Normalize for hashing
        combined = f"{human_text.lower().strip()}{gpt_text.lower().strip()}"
        combined = re.sub(r'\s+', ' ', combined)
        
        return hashlib.md5(combined.encode()).hexdigest()
    
    def validate_conversation(self, conversation: Dict) -> bool:
        """Validate conversation structure and content"""
        try:
            # Check structure
            if "conversations" not in conversation:
                return False
            
            convs = conversation["conversations"]
            if len(convs) != 2:
                return False
                
            if convs[0]["from"] != "human" or convs[1]["from"] != "gpt":
                return False
            
            # Check content quality
            human_text = self.normalize_text(convs[0]["value"])
            gpt_text = self.normalize_text(convs[1]["value"])
            
            if not human_text or not gpt_text:
                return False
                
            # Update normalized content
            conversation["conversations"][0]["value"] = human_text
            conversation["conversations"][1]["value"] = gpt_text
            
            # Validate other fields
            if "expert_id" not in conversation or conversation["expert_id"] not in [0, 1, 2, 3]:
                return False
                
            if "style" not in conversation or conversation["style"] not in ["therapeutic", "educational", "empathetic", "practical"]:
                return False
            
            return True
            
        except Exception:
            return False
    
    def clean_dataset(self, input_file: Path, output_file: Path) -> Dict:
        """Clean and deduplicate dataset"""
        print(f"Cleaning {input_file.name}...")
        
        with open(input_file, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        cleaned_conversations = []
        stats = {
            "original_count": len(conversations),
            "duplicates_removed": 0,
            "invalid_removed": 0,
            "final_count": 0
        }
        
        for conv in conversations:
            # Validate and normalize
            if not self.validate_conversation(conv):
                stats["invalid_removed"] += 1
                continue
            
            # Check for duplicates
            content_hash = self.get_content_hash(conv)
            if content_hash in self.seen_hashes:
                stats["duplicates_removed"] += 1
                continue
            
            self.seen_hashes.add(content_hash)
            cleaned_conversations.append(conv)
        
        stats["final_count"] = len(cleaned_conversations)
        
        # Save cleaned dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_conversations, f, indent=2, ensure_ascii=False)
        
        return stats
    
    def update_config(self, config_file: Path, train_stats: Dict, val_stats: Dict):
        """Update configuration with cleaned stats"""
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Update data config
        config["data_config"]["train_size"] = train_stats["final_count"]
        config["data_config"]["val_size"] = val_stats["final_count"]
        config["data_config"]["total_conversations"] = train_stats["final_count"] + val_stats["final_count"]
        
        # Add cleaning stats
        config["cleaning_stats"] = {
            "original_total": train_stats["original_count"] + val_stats["original_count"],
            "duplicates_removed": train_stats["duplicates_removed"] + val_stats["duplicates_removed"],
            "invalid_removed": train_stats["invalid_removed"] + val_stats["invalid_removed"],
            "final_total": train_stats["final_count"] + val_stats["final_count"],
            "retention_rate": round((train_stats["final_count"] + val_stats["final_count"]) / (train_stats["original_count"] + val_stats["original_count"]) * 100, 2)
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

def main():
    """Clean and deduplicate final dataset"""
    cleaner = DatasetCleaner()
    
    dataset_dir = Path("/root/pixelated/ai/data/lightning_h100_complete")
    
    # Clean training set
    train_file = dataset_dir / "train.json"
    train_clean = dataset_dir / "train_clean.json"
    train_stats = cleaner.clean_dataset(train_file, train_clean)
    
    # Clean validation set
    val_file = dataset_dir / "validation.json"
    val_clean = dataset_dir / "validation_clean.json"
    val_stats = cleaner.clean_dataset(val_file, val_clean)
    
    # Update config
    config_file = dataset_dir / "lightning_config.json"
    cleaner.update_config(config_file, train_stats, val_stats)
    
    # Replace original files
    train_clean.rename(train_file)
    val_clean.rename(val_file)
    
    print(f"\n=== DATASET CLEANING COMPLETE ===")
    print(f"Original total: {train_stats['original_count'] + val_stats['original_count']:,}")
    print(f"Duplicates removed: {train_stats['duplicates_removed'] + val_stats['duplicates_removed']:,}")
    print(f"Invalid removed: {train_stats['invalid_removed'] + val_stats['invalid_removed']:,}")
    print(f"Final total: {train_stats['final_count'] + val_stats['final_count']:,}")
    print(f"Retention rate: {round((train_stats['final_count'] + val_stats['final_count']) / (train_stats['original_count'] + val_stats['original_count']) * 100, 2)}%")
    print(f"\nCleaned dataset ready for Lightning.ai H100 training!")

if __name__ == "__main__":
    main()
