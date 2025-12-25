"""
Enhance Psychology Knowledge Base with xmu_psych_books dataset.

This script adds book reference information from the xmu_psych_books dataset
to the existing psychology knowledge base, enriching it with library collection data.
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

class PsychologyKnowledgeEnhancer:
    """Enhancer for psychology knowledge base with xmu_psych_books data."""

    def __init__(
        self,
        knowledge_base_path: str = "ai/pixel/knowledge/psychology_knowledge_base.json",
        xmu_books_path: str = "ai/training_data_consolidated/xmu_psych_books_processed.jsonl"
    ):
        """Initialize the enhancer with paths to the knowledge base and xmu books data."""
        self.knowledge_base_path = Path(knowledge_base_path)
        self.xmu_books_path = Path(xmu_books_path)
        self.enhanced_knowledge_base: Dict[str, Any] = {}

    def load_knowledge_base(self) -> Dict[str, Any]:
        """Load the existing psychology knowledge base."""
        if not self.knowledge_base_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.knowledge_base_path}")

        with open(self.knowledge_base_path, "r", encoding="utf-8") as f:
            knowledge_base = json.load(f)

        logger.info(f"Loaded psychology knowledge base with {len(knowledge_base.get('concepts', {}))} concepts")
        return knowledge_base

    def load_xmu_books_data(self) -> List[Dict[str, Any]]:
        """Load the processed xmu_psych_books data."""
        if not self.xmu_books_path.exists():
            raise FileNotFoundError(f"xmu_psych_books data not found: {self.xmu_books_path}")

        books_data = []
        with open(self.xmu_books_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    books_data.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse line {line_num} in xmu_books data: {e}")
                    continue

        logger.info(f"Loaded {len(books_data)} entries from xmu_psych_books data")
        return books_data

    def extract_book_references(self, books_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract book reference information for psychology-related books."""
        book_references = []

        for entry in books_data:
            metadata = entry.get("metadata", {})

            # Only process psychology-related books
            if metadata.get("category") == "psychology_book_reference":
                messages = entry.get("messages", [])
                if len(messages) >= 2:
                    user_message = messages[0]
                    assistant_message = messages[1]

                    # Extract book information from the assistant response
                    content = assistant_message.get("content", "")

                    # Create a book reference concept
                    book_concept = {
                        "concept_id": f"book_ref_{entry.get('conversation_id', '').split('_')[-1]}",
                        "name": user_message.get("content", "").replace("What is the psychology book titled '", "").replace("' about?", ""),
                        "category": "psychology_book_reference",
                        "definition": content,
                        "source_transcript": "xmu_psych_books_dataset",
                        "expert_source": "Library Collection Reference",
                        "confidence_score": 0.9,
                        "book_metadata": {
                            "title": user_message.get("content", "").replace("What is the psychology book titled '", "").replace("' about?", ""),
                            "author": self._extract_author(content),
                            "isbn": self._extract_isbn(content),
                            "call_number": self._extract_call_number(content)
                        }
                    }

                    book_references.append(book_concept)

        logger.info(f"Extracted {len(book_references)} psychology book references")
        return book_references

    def _extract_author(self, content: str) -> str:
        """Extract author information from content."""
        # Simple extraction - in a real implementation, this would be more sophisticated
        if "by " in content:
            start = content.find("by ") + 3
            end = content.find(".", start)
            if end == -1:
                end = len(content)
            return content[start:end].strip()
        return ""

    def _extract_isbn(self, content: str) -> str:
        """Extract ISBN information from content."""
        if "ISBN " in content:
            start = content.find("ISBN ") + 5
            end = content.find(".", start)
            if end == -1:
                end = len(content)
            return content[start:end].strip()
        return ""

    def _extract_call_number(self, content: str) -> str:
        """Extract call number information from content."""
        if "call number is " in content:
            start = content.find("call number is ") + 15
            end = content.find(".", start)
            if end == -1:
                end = len(content)
            return content[start:end].strip()
        return ""

    def enhance_knowledge_base(self, knowledge_base: Dict[str, Any], book_references: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Enhance the knowledge base with book references."""
        # Create a copy of the knowledge base
        enhanced_kb = json.loads(json.dumps(knowledge_base))

        # Get the concepts dictionary
        concepts = enhanced_kb.get("concepts", {})

        # Add book references to the concepts
        for book_ref in book_references:
            concept_id = book_ref["concept_id"]
            concepts[concept_id] = book_ref

        enhanced_kb["concepts"] = concepts

        # Update metadata
        if "metadata" not in enhanced_kb:
            enhanced_kb["metadata"] = {}
        enhanced_kb["metadata"]["xmu_books_enhanced"] = True
        enhanced_kb["metadata"]["xmu_books_count"] = len(book_references)
        enhanced_kb["metadata"]["enhancement_timestamp"] = "2025-12-25"

        logger.info(f"Enhanced knowledge base with {len(book_references)} book references")
        return enhanced_kb

    def save_enhanced_knowledge_base(self, enhanced_kb: Dict[str, Any], output_path: str) -> None:
        """Save the enhanced knowledge base to a file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(enhanced_kb, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved enhanced knowledge base to {output_file}")

    def enhance(self, output_path: str = "ai/pixel/knowledge/enhanced_psychology_knowledge_base.json") -> None:
        """Main enhancement process."""
        logger.info("Starting psychology knowledge base enhancement with xmu_psych_books data")

        # Load existing knowledge base
        knowledge_base = self.load_knowledge_base()

        # Load xmu books data
        books_data = self.load_xmu_books_data()

        # Extract book references
        book_references = self.extract_book_references(books_data)

        # Enhance knowledge base
        enhanced_kb = self.enhance_knowledge_base(knowledge_base, book_references)

        # Save enhanced knowledge base
        self.save_enhanced_knowledge_base(enhanced_kb, output_path)

        logger.info("Psychology knowledge base enhancement complete")

def main():
    """Main function to enhance the psychology knowledge base."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhance psychology knowledge base with xmu_psych_books data")
    parser.add_argument(
        "--knowledge-base-path",
        default="ai/pixel/knowledge/psychology_knowledge_base.json",
        help="Path to the existing psychology knowledge base"
    )
    parser.add_argument(
        "--xmu-books-path",
        default="ai/training_data_consolidated/xmu_psych_books_processed.jsonl",
        help="Path to the processed xmu_psych_books data"
    )
    parser.add_argument(
        "--output-path",
        default="ai/pixel/knowledge/enhanced_psychology_knowledge_base.json",
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
        enhancer = PsychologyKnowledgeEnhancer(
            knowledge_base_path=args.knowledge_base_path,
            xmu_books_path=args.xmu_books_path
        )
        enhancer.enhance(output_path=args.output_path)

        print(f"Successfully enhanced psychology knowledge base")
        print(f"Output saved to: {args.output_path}")

    except Exception as e:
        logger.error(f"Error enhancing knowledge base: {e}")
        raise

if __name__ == "__main__":
    main()