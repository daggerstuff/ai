#!/usr/bin/env python3
"""
Lightning.ai H100 LoRA Format Converter
Converts training pairs to Lightning.ai compatible format with expert routing
"""

import json
import random
from pathlib import Path


class LoRAFormatConverter:
    def __init__(self):
        # Expert mapping for MoE architecture
        self.expert_mapping = {"therapeutic": 0, "educational": 1, "empathetic": 2, "practical": 3}

        # Lightning.ai conversation format template
        self.conversation_template = {
            "conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}],
            "expert_id": 0,
            "style": "",
            "quality": 0.0,
            "source": "",
            "metadata": {},
        }

    def format_conversation(self, training_pair: dict) -> dict:
        """Convert training pair to Lightning.ai conversation format"""
        return {
            "conversations": [
                {"from": "human", "value": training_pair["input"]},
                {"from": "gpt", "value": training_pair["output"]},
            ],
            "expert_id": self.expert_mapping[training_pair["style"]],
            "style": training_pair["style"],
            "quality": training_pair["quality"],
            "source": training_pair["source"],
            "metadata": {
                "confidence": training_pair["confidence"],
                "file": training_pair["file"],
                "expert_name": training_pair["style"],
            },
        }

    def create_training_split(self, conversations: list[dict], train_ratio: float = 0.9) -> tuple:
        """Split conversations into training and validation sets"""
        random.shuffle(conversations)
        split_idx = int(len(conversations) * train_ratio)

        train_set = conversations[:split_idx]
        val_set = conversations[split_idx:]

        return train_set, val_set

    def process_training_files(self, input_dir: Path, output_dir: Path) -> dict:
        """Process all training pair files and create Lightning.ai format"""
        output_dir.mkdir(exist_ok=True)

        all_conversations = []
        stats = {
            "total_pairs": 0,
            "by_style": {"therapeutic": 0, "educational": 0, "empathetic": 0, "practical": 0},
            "by_quality": {"high": 0, "medium": 0},
        }

        # Process each training file
        for training_file in input_dir.glob("training_*.json"):
            with open(training_file, encoding="utf-8") as f:
                training_pairs = json.load(f)

            for pair in training_pairs:
                conversation = self.format_conversation(pair)
                all_conversations.append(conversation)

                # Update stats
                stats["total_pairs"] += 1
                stats["by_style"][pair["style"]] += 1

                quality_level = "high" if pair["quality"] >= 0.75 else "medium"
                stats["by_quality"][quality_level] += 1

        # Create train/validation split
        train_conversations, val_conversations = self.create_training_split(all_conversations)

        # Save training set
        train_file = output_dir / "train.json"
        with open(train_file, "w", encoding="utf-8") as f:
            json.dump(train_conversations, f, indent=2, ensure_ascii=False)

        # Save validation set
        val_file = output_dir / "validation.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val_conversations, f, indent=2, ensure_ascii=False)

        # Create expert-specific datasets for analysis
        expert_conversations = {style: [] for style in self.expert_mapping}
        for conv in all_conversations:
            expert_conversations[conv["style"]].append(conv)

        for style, convs in expert_conversations.items():
            expert_file = output_dir / f"expert_{style}.json"
            with open(expert_file, "w", encoding="utf-8") as f:
                json.dump(convs, f, indent=2, ensure_ascii=False)

        # Update stats with split info
        stats["train_size"] = len(train_conversations)
        stats["val_size"] = len(val_conversations)

        return stats

    def create_config_file(self, output_dir: Path, stats: dict):
        """Create Lightning.ai training configuration"""
        config = {
            "model_config": {
                "base_model": "microsoft/DialoGPT-medium",
                "lora_config": {
                    "r": 16,
                    "lora_alpha": 32,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                    "lora_dropout": 0.05,
                    "bias": "none",
                },
            },
            "training_config": {
                "num_experts": 4,
                "expert_mapping": self.expert_mapping,
                "batch_size": 8,
                "learning_rate": 5e-4,
                "num_epochs": 3,
                "warmup_steps": 100,
                "max_length": 1024,
                "gradient_accumulation_steps": 4,
            },
            "data_config": {
                "train_file": "train.json",
                "validation_file": "validation.json",
                "total_conversations": stats["total_pairs"],
                "train_size": stats["train_size"],
                "val_size": stats["val_size"],
                "style_distribution": stats["by_style"],
                "quality_distribution": stats["by_quality"],
            },
            "expert_config": {
                "therapeutic": {
                    "expert_id": 0,
                    "description": "Handles trauma, healing, and therapeutic guidance",
                    "count": stats["by_style"]["therapeutic"],
                },
                "educational": {
                    "expert_id": 1,
                    "description": "Provides clinical explanations and educational content",
                    "count": stats["by_style"]["educational"],
                },
                "empathetic": {
                    "expert_id": 2,
                    "description": "Offers emotional support and validation",
                    "count": stats["by_style"]["empathetic"],
                },
                "practical": {
                    "expert_id": 3,
                    "description": "Gives actionable advice and practical strategies",
                    "count": stats["by_style"]["practical"],
                },
            },
        }

        config_file = output_dir / "lightning_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        return config


def main():
    """Convert training pairs to Lightning.ai H100 LoRA format"""
    converter = LoRAFormatConverter()

    input_dir = Path("/root/pixelated/ai/data/lora_training")
    output_dir = Path("/root/pixelated/ai/data/lightning_h100")

    # Process all training files
    stats = converter.process_training_files(input_dir, output_dir)

    # Create configuration file
    converter.create_config_file(output_dir, stats)

    for _style, _count in stats["by_style"].items():
        pass
    for _quality, _count in stats["by_quality"].items():
        pass


if __name__ == "__main__":
    main()
