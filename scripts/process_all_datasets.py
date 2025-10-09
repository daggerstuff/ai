#!/usr/bin/env python3
"""
Comprehensive processor for all training datasets:
1. Current segments (2,895) - already processed
2. Pixelated-v2 enhanced (61,548) - needs conversion
3. Pixelated-training raw datasets - needs processing
"""

import json
from pathlib import Path
import sys
sys.path.append('/root/pixelated/ai/scripts')
from prompt_generator import TherapeuticPromptGenerator
from lora_format_converter import LoRAFormatConverter

class ComprehensiveDatasetProcessor:
    def __init__(self):
        self.generator = TherapeuticPromptGenerator()
        self.converter = LoRAFormatConverter()
        
    def normalize_segment(self, segment: dict, source: str) -> dict:
        """Normalize segment format across different sources"""
        normalized = {
            "text": segment.get("text", ""),
            "style": segment.get("style", "therapeutic"),
            "confidence": segment.get("confidence", 1.0),
            "quality": segment.get("quality", segment.get("quality_score", 0.7)),
            "source": source,
            "file": segment.get("file", "unknown")
        }
        return normalized
    
    def process_v2_enhanced_data(self):
        """Process 61,548 segments from pixelated-v2"""
        v2_dir = Path("/root/pixelated/ai/lightning/pixelated-v2/training_data/enhanced")
        output_dir = Path("/root/pixelated/ai/data/training_segments_v2")
        output_dir.mkdir(exist_ok=True)
        
        total_processed = 0
        
        for enhanced_file in v2_dir.glob("*.json"):
            if "summary" in enhanced_file.name:
                continue
                
            print(f"Processing v2: {enhanced_file.name}...")
            
            with open(enhanced_file, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            
            training_pairs = []
            for segment in segments:
                try:
                    # Normalize segment format
                    normalized_segment = self.normalize_segment(segment, "pixelated_v2")
                    
                    # Create training pair
                    training_pair = self.generator.create_training_pair(normalized_segment)
                    training_pairs.append(training_pair)
                    total_processed += 1
                except Exception as e:
                    print(f"Error processing v2 segment: {e}")
            
            # Save converted pairs
            output_file = output_dir / f"v2_{enhanced_file.name}"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(training_pairs, f, indent=2, ensure_ascii=False)
            
            print(f"  Converted: {len(training_pairs)} segments")
        
        return total_processed
    
    def classify_style(self, text: str) -> str:
        """Basic style classification for raw datasets"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['trauma', 'heal', 'therapy', 'recover', 'therapeutic']):
            return 'therapeutic'
        elif any(word in text_lower for word in ['understand', 'explain', 'learn', 'study', 'education']):
            return 'educational'
        elif any(word in text_lower for word in ['feel', 'hurt', 'pain', 'support', 'empathy']):
            return 'empathetic'
        else:
            return 'practical'
    
    def process_raw_conversations(self):
        """Process raw conversation datasets from pixelated-training"""
        raw_dir = Path("/root/pixelated/ai/lightning/pixelated-training/processed")
        output_dir = Path("/root/pixelated/ai/data/training_segments_raw")
        output_dir.mkdir(exist_ok=True)
        
        total_segments = 0
        
        # Process filtered datasets
        filtered_dir = raw_dir / "filtered_datasets"
        if filtered_dir.exists():
            for dataset_file in filtered_dir.glob("*.json"):
                print(f"Processing raw: {dataset_file.name}...")
                
                with open(dataset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                segments = []
                
                # Handle different dataset structures
                if "filtered_conversations" in data:
                    conversations = data["filtered_conversations"]
                elif isinstance(data, list):
                    conversations = data
                else:
                    conversations = [data]
                
                # Extract segments from conversations
                for conv in conversations:
                    if "messages" in conv:
                        for msg in conv["messages"]:
                            if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                                segment = {
                                    "text": msg["content"],
                                    "style": self.classify_style(msg["content"]),
                                    "confidence": 1.0,
                                    "quality": 0.7,
                                    "source": "pixelated_training",
                                    "file": dataset_file.name
                                }
                                segments.append(segment)
                
                # Convert to training pairs
                training_pairs = []
                for segment in segments:
                    try:
                        training_pair = self.generator.create_training_pair(segment)
                        training_pairs.append(training_pair)
                        total_segments += 1
                    except Exception as e:
                        print(f"Error: {e}")
                
                if training_pairs:
                    output_file = output_dir / f"raw_{dataset_file.name}"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Converted: {len(training_pairs)} segments")
        
        # Process natural conversations
        natural_file = raw_dir / "natural_conversations" / "natural_multi_turn_conversations.json"
        if natural_file.exists():
            print(f"Processing raw: {natural_file.name}...")
            
            with open(natural_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            conversations = data.get("natural_conversations", [])
            
            for conv in conversations:
                if "messages" in conv:
                    for msg in conv["messages"]:
                        if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                            segment = {
                                "text": msg["content"],
                                "style": self.classify_style(msg["content"]),
                                "confidence": 1.0,
                                "quality": 0.7,
                                "source": "natural_conversations",
                                "file": "natural_multi_turn_conversations.json"
                            }
                            segments.append(segment)
            
            training_pairs = []
            for segment in segments:
                try:
                    training_pair = self.generator.create_training_pair(segment)
                    training_pairs.append(training_pair)
                    total_segments += 1
                except Exception as e:
                    print(f"Error: {e}")
            
            if training_pairs:
                output_file = output_dir / "raw_natural_conversations.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(training_pairs, f, indent=2, ensure_ascii=False)
                
                print(f"  Converted: {len(training_pairs)} segments")
        
        return total_segments
    
    def merge_all_datasets(self):
        """Merge all processed datasets into final Lightning.ai format"""
        all_training_dirs = [
            Path("/root/pixelated/ai/data/lora_training"),      # Original 2,895
            Path("/root/pixelated/ai/data/training_segments_v2"), # V2 enhanced
            Path("/root/pixelated/ai/data/training_segments_raw") # Raw datasets
        ]
        
        output_dir = Path("/root/pixelated/ai/data/lightning_h100_complete")
        output_dir.mkdir(exist_ok=True)
        
        all_conversations = []
        total_stats = {
            "total_pairs": 0,
            "by_style": {"therapeutic": 0, "educational": 0, "empathetic": 0, "practical": 0},
            "by_quality": {"high": 0, "medium": 0},
            "by_source": {}
        }
        
        # Process all training directories
        for training_dir in all_training_dirs:
            if not training_dir.exists():
                continue
                
            print(f"Merging from: {training_dir}")
            
            for training_file in training_dir.glob("*.json"):
                if "summary" in training_file.name:
                    continue
                    
                with open(training_file, 'r', encoding='utf-8') as f:
                    training_pairs = json.load(f)
                
                for pair in training_pairs:
                    conversation = self.converter.format_conversation(pair)
                    all_conversations.append(conversation)
                    
                    # Update stats
                    total_stats["total_pairs"] += 1
                    total_stats["by_style"][pair["style"]] += 1
                    
                    # Track quality
                    quality_level = "high" if pair.get("quality", 0.7) >= 0.75 else "medium"
                    total_stats["by_quality"][quality_level] += 1
                    
                    source = pair.get("source", "unknown")
                    total_stats["by_source"][source] = total_stats["by_source"].get(source, 0) + 1
        
        # Create train/validation split
        train_conversations, val_conversations = self.converter.create_training_split(all_conversations)
        
        # Save final datasets
        train_file = output_dir / "train.json"
        with open(train_file, 'w', encoding='utf-8') as f:
            json.dump(train_conversations, f, indent=2, ensure_ascii=False)
        
        val_file = output_dir / "validation.json"
        with open(val_file, 'w', encoding='utf-8') as f:
            json.dump(val_conversations, f, indent=2, ensure_ascii=False)
        
        # Update stats
        total_stats["train_size"] = len(train_conversations)
        total_stats["val_size"] = len(val_conversations)
        
        # Create final config
        config = self.converter.create_config_file(output_dir, total_stats)
        
        return total_stats

def main():
    """Process all datasets and create comprehensive training data"""
    processor = ComprehensiveDatasetProcessor()
    
    print("=== COMPREHENSIVE DATASET PROCESSING ===")
    
    # Process v2 enhanced data (61,548 segments)
    print("\n1. Processing pixelated-v2 enhanced data...")
    v2_count = processor.process_v2_enhanced_data()
    print(f"V2 segments processed: {v2_count}")
    
    # Process raw conversation datasets
    print("\n2. Processing raw conversation datasets...")
    raw_count = processor.process_raw_conversations()
    print(f"Raw segments processed: {raw_count}")
    
    # Merge all datasets
    print("\n3. Merging all datasets...")
    final_stats = processor.merge_all_datasets()
    
    print(f"\n=== FINAL COMPREHENSIVE DATASET ===")
    print(f"Total conversations: {final_stats['total_pairs']}")
    print(f"Training set: {final_stats['train_size']}")
    print(f"Validation set: {final_stats['val_size']}")
    print(f"\nStyle distribution:")
    for style, count in final_stats['by_style'].items():
        print(f"  {style}: {count}")
    print(f"\nSource distribution:")
    for source, count in final_stats['by_source'].items():
        print(f"  {source}: {count}")
    print(f"\nFinal dataset: /root/pixelated/ai/data/lightning_h100_complete/")

if __name__ == "__main__":
    main()
