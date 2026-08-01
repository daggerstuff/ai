import json
import logging
import os
import random
import sys

from dataset_pipeline.extractors.book_extractor import BookExtractor
from dataset_pipeline.extractors.dataset_loader import DatasetLoader
from dataset_pipeline.extractors.s3_streamer import S3Streamer
from dataset_pipeline.extractors.voice_extractor import VoiceExtractor
from dataset_pipeline.processors.chatml_converter import ChatMLConverter
from dataset_pipeline.processors.quality_filter import QualityFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ratios():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "dataset_ratios.json")
    with open(config_path) as f:
        return json.load(f)

def main():
    logger.info("Starting S3 canonical dataset compilation with upgraded QualityFilter...")
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

    stats = {cat: {"extracted": 0, "accepted": 0, "rejected": 0} for cat in dataset_buckets}

    logger.info("Extracting voice training data...")
    for record in voice_ext.extract_all():
        stats["voice_training"]["extracted"] += 1
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["voice_training"].append(chatml)
            stats["voice_training"]["accepted"] += 1
        else:
            stats["voice_training"]["rejected"] += 1

    logger.info("Extracting psychology knowledge books...")
    for record in book_ext.extract_all():
        stats["psychology_knowledge"]["extracted"] += 1
        chatml = converter.convert(record)
        if quality.passes_filter(chatml):
            dataset_buckets["psychology_knowledge"].append(chatml)
            stats["psychology_knowledge"]["accepted"] += 1
        else:
            stats["psychology_knowledge"]["rejected"] += 1

    # Mental Health Conversations (Priority 1-3 + Consolidated + Stage 1/2)
    logger.info("Extracting mental health conversation datasets...")
    mh_files = [
        "datasets/consolidated/datasets/priority_1_FINAL.jsonl",
        "datasets/consolidated/datasets/priority_2_FINAL.jsonl",
        "datasets/consolidated/datasets/priority_3_FINAL.jsonl",
        "datasets/consolidated/final_datasets/unified_training_data.jsonl",
        "archive/gdrive/tier1_priority/merged_mental_health_dataset.jsonl",
        "datasets/training_v3/stage1_foundation/Amod_mental_health_counseling_conversations.jsonl",
        "datasets/training_v3/stage1_foundation/heliosbrahma_mental_health_chatbot_dataset.jsonl",
        "datasets/training_v3/stage2_specialist_addiction/fadodr_mental_health_therapy.jsonl",
        "datasets/training_v3/stage2_specialist_personality/Kanakmi_mental-disorders.jsonl"
    ]
    for f in mh_files:
        for record in data_ext.load_jsonl(f, "mental_health_conversations", "mental_health"):
            stats["mental_health_conversations"]["extracted"] += 1
            chatml = converter.convert(record)
            if quality.passes_filter(chatml):
                dataset_buckets["mental_health_conversations"].append(chatml)
                stats["mental_health_conversations"]["accepted"] += 1
            else:
                stats["mental_health_conversations"]["rejected"] += 1

    # Reasoning Enhancement (cot_reasoning + filtered CoT)
    logger.info("Extracting reasoning enhancement datasets...")
    cot_files = [
        "cot_reasoning/",
        "datasets/consolidated/datasets/cot_reasoning_filtered.json"
    ]
    for f in cot_files:
        for record in data_ext.load_jsonl(f, "reasoning_enhancement", "cot"):
            stats["reasoning_enhancement"]["extracted"] += 1
            chatml = converter.convert(record)
            if quality.passes_filter(chatml):
                dataset_buckets["reasoning_enhancement"].append(chatml)
                stats["reasoning_enhancement"]["accepted"] += 1
            else:
                stats["reasoning_enhancement"]["rejected"] += 1

    # Psychology Knowledge & Professional Datasets
    logger.info("Extracting psychology knowledge & professional datasets...")
    psych_files = [
        "datasets/consolidated/datasets/psychology_dataset.json",
        "datasets/consolidated/datasets/professional_psychology_filtered.json",
        "datasets/consolidated/datasets/reddit_mental_health_filtered.json"
    ]
    for f in psych_files:
        for record in data_ext.load_jsonl(f, "psychology_knowledge", "psychology"):
            stats["psychology_knowledge"]["extracted"] += 1
            chatml = converter.convert(record)
            if quality.passes_filter(chatml):
                dataset_buckets["psychology_knowledge"].append(chatml)
                stats["psychology_knowledge"]["accepted"] += 1
            else:
                stats["psychology_knowledge"]["rejected"] += 1

    # Personality Balancing & SoulChat / Neuro
    logger.info("Extracting personality balancing datasets...")
    persona_files = [
        "datasets/training_v3/stage4_voice_persona/",
        "datasets/consolidated/datasets/professional_soulchat_filtered.json",
        "datasets/consolidated/datasets/professional_neuro_filtered.json"
    ]
    for f in persona_files:
        for record in data_ext.load_jsonl(f, "personality_balancing", "synthetic_persona"):
            stats["personality_balancing"]["extracted"] += 1
            chatml = converter.convert(record)
            if quality.passes_filter(chatml):
                dataset_buckets["personality_balancing"].append(chatml)
                stats["personality_balancing"]["accepted"] += 1
            else:
                stats["personality_balancing"]["rejected"] += 1


    logger.info("=== Extraction & Quality Filtering Summary ===")
    for cat, s in stats.items():
        logger.info("Bucket '%s': Extracted=%d, Accepted=%d, Rejected=%d (Dedup/Quality Rate=%.1f%%)",
                    cat, s["extracted"], s["accepted"], s["rejected"],
                    (s["rejected"] / s["extracted"] * 100) if s["extracted"] > 0 else 0)

    target_size = 100000
    final_dataset = []

    for category, ratio in ratios.items():
        bucket_data = dataset_buckets.get(category, [])
        target_count = int(target_size * ratio)

        if len(bucket_data) == 0:
            continue

        if len(bucket_data) > target_count:
            sampled = random.sample(bucket_data, target_count)
        else:
            sampled = bucket_data

        logger.info("Sampled %d items for category '%s' (target %d)", len(sampled), category, target_count)
        final_dataset.extend(sampled)

    random.shuffle(final_dataset)

    output_key = "final_dataset/final_training_dataset.jsonl"
    logger.info("Writing clean, deduplicated dataset of %d items to S3 key: %s...", len(final_dataset), output_key)

    # Write to S3
    streamer.write_jsonl(output_key, final_dataset)
    logger.info("Compilation complete! Canonical S3 dataset successfully updated.")


if __name__ == "__main__":
    main()

