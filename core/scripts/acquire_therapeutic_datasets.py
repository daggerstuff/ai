#!/usr/bin/env python3
"""
Acquire Real Therapeutic Conversation Datasets
Downloads and prepares high-quality therapeutic conversation datasets from HuggingFace and other sources
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TherapeuticDatasetAcquisition:
    """Acquire and prepare therapeutic conversation datasets"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("ai/data/acquired_datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define high-quality therapeutic conversation datasets
        self.datasets = {
            "mental_health_counseling": {
                "source": "Amod/mental_health_counseling_conversations",
                "type": "huggingface",
                "description": "Mental health counseling conversations",
                "priority": 1
            },
            "psych_101": {
                "source": "marcelbinz/Psych-101",
                "type": "huggingface",
                "description": "Psychology educational dataset",
                "priority": 2
            },
            "empathetic_dialogues": {
                "source": "empathetic_dialogues",
                "type": "huggingface",
                "description": "Empathetic conversation dataset",
                "priority": 1
            },
            "counselchat": {
                "source": "nbertagnolli/counsel-chat",
                "type": "huggingface",
                "description": "Mental health counseling Q&A",
                "priority": 2
            },
        }

    def download_huggingface_dataset(self, dataset_name: str, config: Dict) -> List[Dict]:
        """Download dataset from HuggingFace"""
        logger.info(f"📥 Downloading {dataset_name} from HuggingFace...")

        try:
            dataset = load_dataset(config["source"], split="train")

            conversations = []
            for item in tqdm(dataset, desc=f"Processing {dataset_name}"):
                # Convert to standard format
                conv = self._convert_to_standard_format(item, dataset_name)
                if conv:
                    conversations.append(conv)

            logger.info(f"✅ Downloaded {len(conversations)} conversations from {dataset_name}")
            return conversations

        except Exception as e:
            logger.error(f"❌ Error downloading {dataset_name}: {e}")
            return []

    def _convert_to_standard_format(self, item: Dict, source: str) -> Optional[Dict]:
        """Convert various dataset formats to standard conversation format"""

        # Try to extract conversation based on common patterns
        conversation = []

        # Pattern 1: Direct conversation field
        if "conversation" in item:
            return {
                "conversation": item["conversation"],
                "metadata": {
                    "source": source,
                    "original_format": "conversation"
                }
            }

        # Pattern 2: Question/Answer pairs
        if "Context" in item and "Response" in item:
            conversation = [
                {"role": "client", "content": item["Context"]},
                {"role": "therapist", "content": item["Response"]}
            ]
        elif "input" in item and "response" in item:
            conversation = [
                {"role": "client", "content": item["input"]},
                {"role": "therapist", "content": item["response"]}
            ]
        elif "question" in item and "answer" in item:
            conversation = [
                {"role": "client", "content": item["question"]},
                {"role": "therapist", "content": item["answer"]}
            ]

        if conversation:
            return {
                "conversation": conversation,
                "metadata": {
                    "source": source,
                    "original_item": item
                }
            }

        return None

    def acquire_all_datasets(self) -> Dict[str, List[Dict]]:
        """Acquire all configured datasets"""
        logger.info("🚀 Starting therapeutic dataset acquisition...")

        all_datasets = {}

        # First, acquire local CoT reasoning datasets
        logger.info("📥 Loading local CoT reasoning datasets...")
        cot_data = self._load_cot_datasets()
        if cot_data:
            all_datasets["cot_reasoning"] = cot_data
            output_file = self.output_dir / "cot_reasoning.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cot_data, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Saved cot_reasoning to {output_file}")

        # Then acquire from HuggingFace
        for name, config in self.datasets.items():
            if config["type"] == "huggingface":
                conversations = self.download_huggingface_dataset(name, config)
                if conversations:
                    all_datasets[name] = conversations

                    # Save individual dataset
                    output_file = self.output_dir / f"{name}.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(conversations, f, indent=2, ensure_ascii=False)
                    logger.info(f"💾 Saved {name} to {output_file}")

        # Create summary
        summary = {
            "total_datasets": len(all_datasets),
            "total_conversations": sum(len(convs) for convs in all_datasets.values()),
            "datasets": {
                name: {
                    "count": len(convs),
                    "source": self.datasets.get(name, {}).get("source", "local") if name != "cot_reasoning" else "local_cot_datasets"
                }
                for name, convs in all_datasets.items()
            }
        }

        summary_file = self.output_dir / "acquisition_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n📊 Acquisition Summary:")
        logger.info(f"   Total datasets: {summary['total_datasets']}")
        logger.info(f"   Total conversations: {summary['total_conversations']}")
        for name, info in summary['datasets'].items():
            logger.info(f"   - {name}: {info['count']} conversations")
        logger.info(f"   Summary saved to: {summary_file}")

        return all_datasets

    def _load_cot_datasets(self) -> List[Dict]:
        """Load Chain of Thought reasoning datasets from local files"""
        cot_conversations = []

        # Load filtered CoT reasoning dataset
        cot_file = Path("ai/training_data_consolidated/datasets/cot_reasoning_filtered.json")
        if cot_file.exists():
            try:
                with open(cot_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if "filtered_conversations" in data:
                        for item in data["filtered_conversations"]:
                            # Convert to standard format
                            conversation = []
                            for msg in item.get("messages", []):
                                role = "client" if msg["role"] == "user" else "therapist"
                                conversation.append({"role": role, "content": msg["content"]})

                            if conversation:
                                cot_conversations.append({
                                    "conversation": conversation,
                                    "metadata": {
                                        "source": "cot_reasoning",
                                        "dataset": "cot_reasoning",
                                        "quality_level": item.get("metadata", {}).get("quality_level", "unknown"),
                                        "original_id": item.get("id")
                                    }
                                })
                logger.info(f"✅ Loaded {len(cot_conversations)} CoT reasoning conversations")
            except Exception as e:
                logger.error(f"❌ Error loading CoT dataset: {e}")

        return cot_conversations


def main():
    """Main execution"""
    acquisitor = TherapeuticDatasetAcquisition()
    datasets = acquisitor.acquire_all_datasets()

    logger.info("\n✅ Dataset acquisition complete!")
    logger.info(f"📁 Datasets saved to: {acquisitor.output_dir}")

    return datasets


if __name__ == "__main__":
    main()
