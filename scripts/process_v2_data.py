#!/usr/bin/env python3
"""
Process pixelated-v2 enhanced training data (61,548 segments)
Convert to our training format and merge with existing data
"""

import json
from pathlib import Path
import sys
sys.path.append('/root/pixelated/ai/scripts')
from prompt_generator import TherapeuticPromptGenerator
from lora_format_converter import LoRAFormatConverter

def process_v2_enhanced_data():
    """Process the 61,548 segments from pixelated-v2"""
    v2_dir = Path("/root/pixelated/ai/lightning/pixelated-v2/training_data/enhanced")
    output_dir = Path("/root/pixelated/ai/data/training_segments_v2")
    output_dir.mkdir(exist_ok=True)
    
    generator = TherapeuticPromptGenerator()
    total_processed = 0
    
    # Process each enhanced file
    for enhanced_file in v2_dir.glob("*.json"):
        if "summary" in enhanced_file.name:
            continue
            
        print(f"Processing {enhanced_file.name}...")
        
        with open(enhanced_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # Convert to training pairs
        training_pairs = []
        for segment in segments:
            try:
                training_pair = generator.create_training_pair(segment)
                training_pairs.append(training_pair)
                total_processed += 1
            except Exception as e:
                print(f"Error processing segment: {e}")
        
        # Save converted pairs
        output_file = output_dir / f"v2_{enhanced_file.name}"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"  Converted: {len(training_pairs)} segments")
    
    print(f"\nTotal v2 segments processed: {total_processed}")
    return total_processed

if __name__ == "__main__":
    process_v2_enhanced_data()
