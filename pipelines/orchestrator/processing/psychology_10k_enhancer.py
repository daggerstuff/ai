"""
Enhance Psychology Knowledge Base with psychology-10k dataset.

This script adds therapeutic conversation examples from the psychology-10k dataset
to the existing psychology knowledge base.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

class Psychology10kEnhancer:
    """Enhancer for psychology knowledge base with psychology-10k data."""

    def __init__(
        self,
        knowledge_base_path: str = "ai/models/pixel_core/knowledge/enhanced_psychology_knowledge_base.json",
        psychology_10k_path: str = "ai/training_data_consolidated/psychology_10k_processed.jsonl"
    ):
        """Initialize the enhancer with paths to the knowledge base and psychology-10k data."""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.psychology_10k_path = Path(psychology_10k_path)
        self.enhanced_knowledge_base: Dict[str, Any] = {}

    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load the existing psychology knowledge base."""
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")

        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)

        logger.info(f"Loaded psychology knowledge base with {len(knowledge_base.get('concepts', {}))} concepts")
        return knowledge_base

    def load_psychology_10k_data(self) -> List[Dict[str, Any]]:
        """Load the processed psychology-10k data."""
        if not self.psychology_10k_path.exists():
            raise FileNotFoundError(f"psychology-10k data not found: {self.psychology_10k_path}")

        psychology_10k_data = []
        with open(self.psychology_10k_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    psychology_10k_data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in psychology-10k data: {e}")
                    continue

        logger.info(f"Loaded {len(psychology_10k_data)} entries from psychology-10k data")
        return psychology_10k_data

    def extract_therapeutic_examples(self, psychology_10k_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract therapeutic conversation examples for knowledge enhancement."""
        therapeutic_examples = []

        for entry in psychology_10k_data:
            messages = entry.get("messages", [])
            if len(messages) >= 2:
                user_message = messages[0]
                assistant_message = messages[1]

                # Extract the instruction and input from the user message
                content = user_message.get("content", "")

                # Create a therapeutic example concept
                therapeutic_concept = {
                    "concept_id": f"therapeutic_example_{entry.get('conversation_id', '').split('_')[-1]}",
                    "name": f"Therapeutic Conversation Example {entry.get('conversation_id', '').split('_')[-1]}",
                    "category": "therapeutic_conversation_example",
                    "definition": f"Patient: {content}\n\nTherapist: {assistant_message.get('content', '')}",
                    "source_transcript": "psychology_10k_dataset",
                    "expert_source": "Licensed Psychologist Examples",
                    "confidence_score": 0.95,
                    "therapeutic_example_metadata": {
                        "conversation_id": entry.get("conversation_id", ""),
                        "source": entry.get("source", "")
                    }
                }

                therapeutic_examples.append(therapeutic_concept)

        logger.info(f"Extracted {len(therapeutic_examples)} therapeutic examples")
        return therapeutic_examples

    def enhance_knowledge_base(self, knowledge_base: Dict[str, Any], therapeutic_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance the knowledge base with therapeutic examples."""
        # Create a copy of the knowledge base
        enhanced_kb = json.loads(json.dumps(knowledge_base))

        # Get the concepts dictionary
        concepts = enhanced_kb.get("concepts", {})

        # Add therapeutic examples to the concepts
        for therapeutic_example in therapeutic_examples:
            concept_id = therapeutic_example["concept_id"]
            concepts[concept_id] = therapeutic_example

        enhanced_kb["concepts"] = concepts

        # Update metadata
        if "metadata" not in enhanced_kb:
            enhanced_kb["metadata"] = {}
        enhanced_kb["metadata"]["psychology_10k_enhanced"] = True
        enhanced_kb["metadata"]["psychology_10k_count"] = len(therapeutic_examples)
        enhanced_kb["metadata"]["enhancement_timestamp"] = "2025-12-25"

        logger.info(f"Enhanced knowledge base with {len(therapeutic_examples)} therapeutic examples")
        return enhanced_kb

    def save_enhanced_knowledge_base(self, enhanced_kb: Dict[str, Any], output_path: str) -> None:
        """Save the enhanced knowledge base to a file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_kb, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved enhanced knowledge base to {output_file}")

    def enhance(self, output_path: str = "ai/models/pixel_core/knowledge/further_enhanced_psychology_knowledge_base.json") -> None:
        """Main enhancement process."""
        logger.info("Starting psychology knowledge base enhancement with psychology-10k data")

        # Load existing knowledge base
        knowledge_base = self.load_knowledge_base()

        # Load psychology-10k data
        psychology_10k_data = self.load_psychology_10k_data()

        # Extract therapeutic examples
        therapeutic_examples = self.extract_therapeutic_examples(psychology_10k_data)

        # Enhance knowledge base
        enhanced_kb = self.enhance_knowledge_base(knowledge_base, therapeutic_examples)

        # Save enhanced knowledge base
        self.save_enhanced_knowledge_base(enhanced_kb, output_path)

        logger.info("Psychology knowledge base enhancement complete")

def main():
    """Main function to enhance the psychology knowledge base."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhance psychology knowledge base with psychology-10k data")
    parser.add_argument(
        "--knowledge-base-path",
        default="ai/models/pixel_core/knowledge/enhanced_psychology_knowledge_base.json",
        help="Path to the existing psychology knowledge base"
    )
    parser.add_argument(
        "--psychology-10k-path",
        default="ai/training_data_consolidated/psychology_10k_processed.jsonl",
        help="Path to the processed psychology-10k data"
    )
    parser.add_argument(
        "--output-path",
        default="ai/models/pixel_core/knowledge/further_enhanced_psychology_knowledge_base.json",
        help="Path to save the enhanced knowledge base"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Enhance the knowledge base
        enhancer = Psychology10kEnhancer(
            knowledge_base_path=args.knowledge_base_path,
            psychology_10k_path=args.psychology_10k_path
        )
        enhancer.enhance(output_path=args.output_path)

        print(f"Successfully enhanced psychology knowledge base")
        print(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error enhancing knowledge base: {e}")
        raise

if __name__ == "__main__":
    main()