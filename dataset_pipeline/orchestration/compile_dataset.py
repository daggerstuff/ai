import os
import json
import random
from dataset_pipeline.extractors.s3_streamer import S3Streamer
from dataset_pipeline.extractors.voice_extractor import VoiceExtractor
from dataset_pipeline.extractors.book_extractor import BookExtractor
from dataset_pipeline.extractors.dataset_loader import DatasetLoader
from dataset_pipeline.processors.chatml_converter import ChatMLConverter
from dataset_pipeline.processors.quality_filter import QualityFilter

def load_ratios():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset_ratios.json")
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    print("Initializing Pixel LLM Dataset Compiler...")
    streamer = S3Streamer()
    voice_ext = VoiceExtractor(streamer)
    book_ext = BookExtractor(streamer)
    data_ext = DatasetLoader(streamer)
    
    converter = ChatMLConverter()
    quality = QualityFilter()
    ratios = load_ratios()
    
    # Categorized buckets
    dataset_buckets = {
        "psychology_knowledge": [],
        "voice_training": [],
        "mental_health_conversations": [],
        "reasoning_enhancement": [],
        "personality_balancing": []
    }

    print("\n--- Phase 1: Streaming & Extracting Voice Data ---")
    for record in voice_ext.extract_all():
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["voice_training"].append(chatml)
            
    print(f"Collected {len(dataset_buckets['voice_training'])} valid voice training records.")

    print("\n--- Phase 2: Streaming & Extracting Psychology Books ---")
    for record in book_ext.extract_all():
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["psychology_knowledge"].append(chatml)
            
    print(f"Collected {len(dataset_buckets['psychology_knowledge'])} valid psychology knowledge records.")

    print("\n--- Phase 2.5: Streaming External Datasets (Mental Health, Reasoning, Personality) ---")
    
    # Mental Health Conversations (merged_mental_health_dataset + stage1)
    mh_files = [
        "archive/gdrive/tier1_priority/merged_mental_health_dataset.jsonl",
        "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
        "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
        "datasets/training_v3/stage2_specialist_addiction/fadodr_mental_health_therapy.jsonl",
        "datasets/training_v3/stage2_specialist_personality/Kanakmi_mental-disorders.jsonl"
    ]
    for f in mh_files:
        for record in data_ext.load_jsonl(f, "mental_health_conversations", "mental_health"):
            chatml = converter.convert(record)
            if quality.passes_filter(chatml):
                dataset_buckets["mental_health_conversations"].append(chatml)
                
    # Reasoning Enhancement (cot_reasoning)
    for record in data_ext.load_jsonl("cot_reasoning/", "reasoning_enhancement", "cot"):
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["reasoning_enhancement"].append(chatml)
            
    # Personality Balancing (stage4_voice_persona from HF)
    for record in data_ext.load_jsonl("datasets/training_v3/stage4_voice_persona/", "personality_balancing", "synthetic_persona"):
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["personality_balancing"].append(chatml)

    print(f"Collected {len(dataset_buckets['mental_health_conversations'])} mental health records.")
    print(f"Collected {len(dataset_buckets['reasoning_enhancement'])} reasoning records.")
    print(f"Collected {len(dataset_buckets['personality_balancing'])} personality balancing records.")

    print("\n--- Phase 3: Balancing & Shuffling ---")
    target_size = 100000
    final_dataset = []
    
    for category, ratio in ratios.items():
        bucket_data = dataset_buckets.get(category, [])
        target_count = int(target_size * ratio)
        
        if len(bucket_data) == 0:
            print(f"Warning: No data for {category}. Skipping.")
            continue
            
        if len(bucket_data) > target_count:
            sampled = random.sample(bucket_data, target_count)
        else:
            # If we don't have enough, just take everything we have
            sampled = bucket_data
            
        print(f"Category {category}: target={target_count}, actual_used={len(sampled)}")
        final_dataset.extend(sampled)

    print(f"\nFinal dataset size before shuffle: {len(final_dataset)} records.")
    random.shuffle(final_dataset)

    print("\n--- Phase 4: Streaming to S3 ---")
    output_key = "final_dataset/final_training_dataset.jsonl"
    print(f"Uploading to HetznerS3: {output_key}...")
    
    # Write to S3
    streamer.write_jsonl(output_key, final_dataset)
    
    print("Pipeline Execution Complete! 🎉")

if __name__ == "__main__":
    main()
