#!/usr/bin/env python3
"""
Process raw datasets from pixelated-training
Convert conversations to our segment format, then to training pairs
"""

import json
from pathlib import Path
import sys
sys.path.append('/root/pixelated/ai/scripts')
from prompt_generator import TherapeuticPromptGenerator

class RawDatasetProcessor:
    def __init__(self):
        self.generator = TherapeuticPromptGenerator()
    
    def process_conversation_dataset(self, file_path: Path) -> list:
        """Process conversation-based datasets"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        segments = []
        
        # Handle different dataset structures
        if "natural_conversations" in data:
            conversations = data["natural_conversations"]
        elif "filtered_conversations" in data:
            conversations = data["filtered_conversations"]
        else:
            conversations = data if isinstance(data, list) else [data]
        
        for conv in conversations:
            if "messages" in conv:
                # Extract assistant responses as segments
                for msg in conv["messages"]:
                    if msg.get("role") == "assistant" and len(msg.get("content", "")) > 100:
                        segment = {
                            "text": msg["content"],
                            "style": self.classify_style(msg["content"]),
                            "confidence": 1.0,
                            "quality": 0.7,
                            "source": file_path.stem,
                            "file": file_path.name
                        }
                        segments.append(segment)
        
        return segments
    
    def classify_style(self, text: str) -> str:
        """Basic style classification for raw datasets"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['trauma', 'heal', 'therapy', 'recover']):
            return 'therapeutic'
        elif any(word in text_lower for word in ['understand', 'explain', 'learn', 'study']):
            return 'educational'
        elif any(word in text_lower for word in ['feel', 'hurt', 'pain', 'support']):
            return 'empathetic'
        else:
            return 'practical'
    
    def process_all_raw_datasets(self):
        """Process all raw datasets from pixelated-training"""
        raw_dir = Path("/root/pixelated/ai/lightning/pixelated-training/processed")
        output_dir = Path("/root/pixelated/ai/data/training_segments_raw")
        output_dir.mkdir(exist_ok=True)
        
        total_segments = 0
        
        # Process filtered datasets
        filtered_dir = raw_dir / "filtered_datasets"
        if filtered_dir.exists():
            for dataset_file in filtered_dir.glob("*.json"):
                print(f"Processing {dataset_file.name}...")
                segments = self.process_conversation_dataset(dataset_file)
                
                if segments:
                    # Convert to training pairs
                    training_pairs = []
                    for segment in segments:
                        try:
                            training_pair = self.generator.create_training_pair(segment)
                            training_pairs.append(training_pair)
                        except Exception as e:
                            print(f"Error: {e}")
                    
                    # Save
                    output_file = output_dir / f"raw_{dataset_file.name}"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(training_pairs, f, indent=2, ensure_ascii=False)
                    
                    print(f"  Converted: {len(training_pairs)} segments")
                    total_segments += len(training_pairs)
        
        # Process natural conversations
        natural_file = raw_dir / "natural_conversations" / "natural_multi_turn_conversations.json"
        if natural_file.exists():
            print(f"Processing {natural_file.name}...")
            segments = self.process_conversation_dataset(natural_file)
            
            if segments:
                training_pairs = []
                for segment in segments:
                    try:
                        training_pair = self.generator.create_training_pair(segment)
                        training_pairs.append(training_pair)
                    except Exception as e:
                        print(f"Error: {e}")
                
                output_file = output_dir / "raw_natural_conversations.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(training_pairs, f, indent=2, ensure_ascii=False)
                
                print(f"  Converted: {len(training_pairs)} segments")
                total_segments += len(training_pairs)
        
        print(f"\nTotal raw segments processed: {total_segments}")
        return total_segments

if __name__ == "__main__":
    processor = RawDatasetProcessor()
    processor.process_all_raw_datasets()
