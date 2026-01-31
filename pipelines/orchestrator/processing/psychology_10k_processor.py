"""
Process psychology-10k dataset and convert to instruction-following format.

This script converts the psychology-10k dataset into instruction-following examples
that can be used for training the AI model.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Try different import paths for conversation schema
conversation_schema_path = project_root / "ai" / "dataset_pipeline" / "schemas"
sys.path.insert(0, str(conversation_schema_path))

try:
    from conversation_schema import Conversation, Message
except ImportError:
    try:
        from schemas.conversation_schema import Conversation, Message
    except ImportError:
        from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message

logger = logging.getLogger(__name__)

class Psychology10kProcessor:
    """Processor for psychology-10k dataset."""

    def __init__(self, dataset_path: str):
        """Initialize the processor with the dataset path."""
        self.dataset_path = Path(dataset_path)
        self.processed_conversations: List[Conversation] = []

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the psychology-10k dataset from JSONL file."""
        dataset_file = self.dataset_path / "train.jsonl"
        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_file}")

        data = []
        with open(dataset_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(data)} entries from psychology-10k dataset")
        return data

    def convert_to_instruction_format(self, data: List[Dict[str, Any]]) -> List[Conversation]:
        """
        Convert psychology-10k entries to instruction-following conversations.

        Args:
            data: List of entries from psychology-10k dataset

        Returns:
            List of Conversation objects in instruction-following format
        """
        conversations = []

        for i, entry in enumerate(data):
            instruction = entry.get("instruction", "")
            user_input = entry.get("input", "")
            assistant_output = entry.get("output", "")

            # Skip entries without essential content
            if not instruction or not user_input or not assistant_output:
                continue

            # Create conversation
            conversation = Conversation(
                conversation_id=f"psychology_10k_{i}",
                source="psychology_10k",
                messages=[
                    Message(role="user", content=f"{instruction}\n\n{user_input}"),
                    Message(role="assistant", content=assistant_output),
                ],
                metadata={
                    "tier": 6,
                    "quality_threshold": 1.0,
                    "original_entry": entry,
                    "category": "therapeutic_conversation",
                }
            )
            conversations.append(conversation)

        logger.info(f"Converted {len(conversations)} entries to instruction format")
        return conversations

    def process(self) -> List[Conversation]:
        """
        Process the psychology-10k dataset.

        Returns:
            List of processed Conversation objects
        """
        logger.info("Starting psychology-10k dataset processing")

        # Load dataset
        data = self.load_dataset()

        # Convert to instruction format
        conversations = self.convert_to_instruction_format(data)

        self.processed_conversations = conversations
        logger.info(f"Processing complete. Generated {len(conversations)} conversations")

        return conversations

    def save_output(self, output_path: str) -> None:
        """
        Save processed conversations to JSONL file.

        Args:
            output_path: Path to save the output file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for conversation in self.processed_conversations:
                f.write(json.dumps(conversation.to_dict(), ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(self.processed_conversations)} conversations to {output_file}")

def main():
    """Main function to process the psychology-10k dataset."""
    import argparse

    parser = argparse.ArgumentParser(description="Process psychology-10k dataset")
    parser.add_argument(
        "--input-path",
        default="/home/vivi/pixelated/ai/datasets/tier6_knowledge/psychology-10k",
        help="Path to psychology-10k dataset directory"
    )
    parser.add_argument(
        "--output-path",
        default="/home/vivi/pixelated/ai/training_data_consolidated/psychology_10k_processed.jsonl",
        help="Path to save processed output"
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Process the dataset
        processor = Psychology10kProcessor(args.input_path)
        conversations = processor.process()

        # Save output
        processor.save_output(args.output_path)

        print(f"Successfully processed {len(conversations)} conversations")
        print(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    main()