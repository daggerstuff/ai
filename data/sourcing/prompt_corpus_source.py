"""
Prompt Corpus Extraction.

Extracts therapeutic dialogue templates from knowledge base books
to populate the prompt corpus for Stage 3/4 training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

logger = logging.getLogger(__name__)


class PromptCorpusSource:
    """
    Extract prompt templates from knowledge base.

    Sources from:
    - Complex PTSD book (Herman)
    - IFS therapy (Schwartz)
    - Brain Energy (Potash)
    - 20+ therapeutic books in knowledge base

    Usage:
        source = PromptCorpusSource()
        for ref in source.fill_gap(500):
            print(f"Extracted: {ref}")
    """

    # Knowledge base book locations
    KNOWLEDGE_SOURCES = [
        Path("ai/data/knowledge_base/complex_ptsd"),
        Path("ai/data/knowledge_base/IFS"),
        Path("ai/data/knowledge_base/brain_energy"),
        Path("ai/data/knowledge_sources_registry.json"),
    ]

    # Template categories for therapeutic dialogue
    TEMPLATE_CATEGORIES = {
        'validation': [
            "It makes sense that you would feel {emotion} given {situation}.",
            "Anyone in your situation would struggle with this.",
            "Your reaction is understandable given what you've been through.",
        ],
        'grounding': [
            "Let's take a moment to notice where your feet are right now.",
            "Can you name three things you can see in this room?",
            "Let's focus on your breathing for a moment.",
        ],
        'psychoeducation': [
            "What you're experiencing is actually a common response to trauma.",
            "This pattern makes sense from a neurological perspective.",
            "Research shows that {technique} can be helpful for {symptom}.",
        ],
        'cognitive_restructuring': [
            "What evidence do you have for that thought?",
            "Is there another way to look at this situation?",
            "What would you tell a friend who had this thought?",
        ],
        'safety_planning': [
            "When you notice these signs, what helps you feel safe?",
            "Who are people you can reach out to when this happens?",
            "What has worked in the past when you felt this way?",
        ],
    }

    def __init__(self, output_dir: Optional[str | Path] = None):
        """
        Initialize prompt corpus source.

        Args:
            output_dir: Directory for extracted prompts
        """
        self.output_dir = Path(output_dir) if output_dir else Path("ai/data/acquired_datasets/prompt_corpus")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / "therapeutic_prompts.jsonl"

    def extract_from_book(self, book_path: Path) -> Iterator[Dict[str, Any]]:
        """
        Extract therapeutic prompts from a book.

        Args:
            book_path: Path to book file

        Yields:
            Prompt records
        """
        if not book_path.exists():
            return

        try:
            if book_path.suffix == '.json':
                with open(book_path) as f:
                    data = json.load(f)
                    # Process based on structure
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if isinstance(value, str) and len(value) > 50:
                                yield self._create_prompt(key, value, book_path.name)
            elif book_path.suffix in ['.txt', '.md']:
                with open(book_path.open()) as f:
                    content = f.read()
                    # Extract therapeutic statements
                    for line in content.split('\n'):
                        if self._is_therapeutic_statement(line):
                            yield self._create_prompt('extracted', line, book_path.name)

        except Exception as e:
            logger.warning(f"Failed to extract from {book_path}: {e}")

    def _is_therapeutic_statement(self, text: str) -> bool:
        """Check if text looks like a therapeutic statement."""
        text_lower = text.lower()
        therapeutic_indicators = [
            'client', 'therapy', 'therapist', 'healing', 'trauma',
            'feel', 'emotion', 'coping', 'symptom', 'treatment',
        ]
        return any(indicator in text_lower for indicator in therapeutic_indicators) and len(text) > 30

    def _create_prompt(
        self,
        category: str,
        content: str,
        source: str
    ) -> Dict[str, Any]:
        """Create a prompt record."""
        return {
            'prompt': f"Client presents with issue. Therapist response:",
            'response': content,
            'metadata': {
                'category': category,
                'source': f'knowledge_base:{source}',
                'stage': 'stage3_edge_stress_test',
                'quality_profile': 'prompt_corpus',
                'template_type': 'therapeutic_response',
            }
        }

    def generate_templates(self, category: str = None) -> Iterator[Dict[str, Any]]:
        """
        Generate prompts from template library.

        Args:
            category: Specific template category or None for all

        Yields:
            Prompt records
        """
        categories = [category] if category else list(self.TEMPLATE_CATEGORIES.keys())

        for cat in categories:
            templates = self.TEMPLATE_CATEGORIES.get(cat, [])
            for template in templates:
                yield self._create_prompt(f"template_{cat}", template, "template_library")

    def fill_gap(self, gap: int, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Extract/generate prompts to fill gap.

        Args:
            gap: Number of samples needed

        Yields:
            Prompt records
        """
        logger.info(f"PromptCorpusSource filling gap of {gap} samples")

        count = 0

        # First, try to extract from knowledge base
        for source_path in self.KNOWLEDGE_SOURCES:
            if count >= gap:
                break

            if source_path.is_dir():
                for book in source_path.glob("*"):
                    if count >= gap:
                        break
                    for prompt in self.extract_from_book(book):
                        yield prompt
                        count += 1
            elif source_path.exists():
                for prompt in self.extract_from_book(source_path):
                    yield prompt
                    count += 1
                    if count >= gap:
                        break

        # Fill remaining with templates
        while count < gap:
            for template_prompt in self.generate_templates():
                yield template_prompt
                count += 1
                if count >= gap:
                    break

        logger.info(f"Prompt corpus sourcing complete: {count} samples")

    def discover(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """Discover available knowledge sources."""
        for source_path in self.KNOWLEDGE_SOURCES:
            if source_path.exists():
                yield {
                    'source_path': str(source_path),
                    'type': 'knowledge_base',
                    'stage': 'stage3_edge_stress_test',
                    'quality_profile': 'prompt_corpus',
                }

        # Template categories
        for category, templates in self.TEMPLATE_CATEGORIES.items():
            yield {
                'category': f'template_{category}',
                'template_count': len(templates),
                'type': 'template_library',
                'stage': 'stage3_edge_stress_test',
                'quality_profile': 'prompt_corpus',
            }


if __name__ == "__main__":
    # Test prompt extraction
    source = PromptCorpusSource()

    print("Discovering prompt sources...")
    for info in source.discover():
        print(f"  {info['type']}: {info.get('category', info.get('source_path', 'unknown'))}")

    print("\nGenerating prompt samples...")
    count = 0
    for record in source.fill_gap(10):
        meta = record.get('metadata', {})
        print(f"  [{meta.get('category', 'unknown')}] {record.get('response', '')[:50]}...")
        count += 1

    print(f"\nGenerated {count} prompt samples")
