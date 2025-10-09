#!/usr/bin/env python3
"""
Rebuild the entire dataset with proper contextual Q/A matching
"""

import json
from pathlib import Path
import sys
sys.path.append('/root/pixelated/ai/scripts')
from contextual_prompt_generator import ContextualPromptGenerator
from lora_format_converter import LoRAFormatConverter

def rebuild_dataset():
    """Rebuild dataset with contextual Q/A pairs"""
    generator = ContextualPromptGenerator()
    converter = LoRAFormatConverter()
    
    # Source directories
    segment_dirs = [
        Path("/root/pixelated/ai/data/training_segments"),
        Path("/root/pixelated/ai/data/training_segments_v2"), 
        Path("/root/pixelated/ai/data/training_segments_raw")
    ]
    
    output_dir = Path("/root/pixelated/ai/data/lightning_h100_contextual")
    output_dir.mkdir(exist_ok=True)
    
    all_conversations = []
    processed_count = 0
    
    print("Rebuilding dataset with contextual Q/A pairs...")
    
    # Process all segment files
    for segment_dir in segment_dirs:
        if not segment_dir.exists():
            continue
            
        print(f"Processing: {segment_dir}")
        
        for segment_file in segment_dir.glob("*.json"):
            if "summary" in segment_file.name:
                continue
                
            with open(segment_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both segment format and training pair format
            if isinstance(data, list) and len(data) > 0:
                if "input" in data[0]:  # Already training pairs
                    segments = [{"text": item["output"], "style": item["style"], 
                               "confidence": item["confidence"], "quality": item["quality"],
                               "source": item["source"], "file": item["file"]} for item in data]
                else:  # Raw segments
                    segments = data
            else:
                continue
            
            # Create contextual training pairs
            for segment in segments:
                try:
                    training_pair = generator.create_training_pair(segment)
                    conversation = converter.format_conversation(training_pair)
                    all_conversations.append(conversation)
                    processed_count += 1
                    
                    if processed_count % 5000 == 0:
                        print(f"  Processed: {processed_count:,} segments")
                        
                except Exception as e:
                    print(f"Error processing segment: {e}")
    
    print(f"Total processed: {processed_count:,} segments")
    
    # Create train/validation split
    train_conversations, val_conversations = converter.create_training_split(all_conversations)
    
    # Save datasets
    train_file = output_dir / "train.json"
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_conversations, f, indent=2, ensure_ascii=False)
    
    val_file = output_dir / "validation.json"  
    with open(val_file, 'w', encoding='utf-8') as f:
        json.dump(val_conversations, f, indent=2, ensure_ascii=False)
    
    # Create stats and config
    stats = {
        "total_pairs": len(all_conversations),
        "train_size": len(train_conversations),
        "val_size": len(val_conversations),
        "by_style": {"therapeutic": 0, "educational": 0, "empathetic": 0, "practical": 0},
        "by_quality": {"high": 0, "medium": 0}
    }
    
    for conv in all_conversations:
        stats["by_style"][conv["style"]] += 1
        quality_level = "high" if conv.get("quality", 0.7) >= 0.75 else "medium"
        stats["by_quality"][quality_level] += 1
    
    config = converter.create_config_file(output_dir, stats)
    
    print(f"\nContextual dataset created:")
    print(f"  Total: {stats['total_pairs']:,}")
    print(f"  Train: {stats['train_size']:,}")
    print(f"  Validation: {stats['val_size']:,}")
    print(f"  Output: {output_dir}")

if __name__ == "__main__":
    rebuild_dataset()
