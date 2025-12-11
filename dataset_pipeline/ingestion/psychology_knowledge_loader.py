#!/usr/bin/env python3
"""
Psychology Knowledge Base Loader for Training Pipeline Integration
Loads 4,867+ psychology concepts, DSM-5 definitions, and therapeutic techniques
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from ai.dataset_pipeline.utils.logger import get_logger

logger = get_logger("dataset_pipeline.psychology_knowledge_loader")


@dataclass
class PsychologyConcept:
    """Structured psychology knowledge concept"""

    concept_id: str
    title: str
    content: str
    category: str
    subcategory: str | None = None
    related_concepts: list[str] = None
    therapeutic_approaches: list[str] = None
    source: str = "psychology_knowledge"

    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []
        if self.therapeutic_approaches is None:
            self.therapeutic_approaches = []

    def to_training_format(self) -> dict:
        """Convert to standard training format"""
        # Create Q&A format for knowledge
        prompt = f"Explain {self.title} in a therapeutic context."
        response = self.content

        # Create conversational text
        text = f"Question: {prompt}\nAnswer: {response}"

        return {
            "text": text,
            "prompt": prompt,
            "response": response,
            "metadata": {
                "source": "psychology_knowledge",
                "concept_id": self.concept_id,
                "category": self.category,
                "subcategory": self.subcategory,
                "related_concepts": self.related_concepts,
                "therapeutic_approaches": self.therapeutic_approaches,
                "is_knowledge_base": True,
                "is_edge_case": False,
            },
        }


@dataclass
class PsychologyKnowledgeConfig:
    """Configuration for PsychologyKnowledgeLoader"""

    knowledge_base_paths: list[Path] = field(
        default_factory=lambda: [
            Path("ai/pixel/knowledge/psychology_knowledge_base_optimized.json"),
            Path("ai/training_data_consolidated/psychology_knowledge_base_optimized.json"),
            Path("ai/dataset_pipeline/data/psychology_knowledge_base.json"),
        ]
    )


class PsychologyKnowledgeLoader:
    """Loader for psychology knowledge base"""

    def __init__(
        self, config: PsychologyKnowledgeConfig | None = None, file_path: Path | None = None
    ):
        self.config = config or PsychologyKnowledgeConfig()

        # Use provided file path or find it in default locations
        if file_path:
            path = Path(file_path)
            if path.is_dir():
                candidates = [
                    path / "psychology_knowledge_base_optimized.json",
                    path / "psychology_knowledge_base.json",
                ]
                self.knowledge_file = candidates[0]
                for candidate in candidates:
                    if candidate.exists():
                        self.knowledge_file = candidate
                        break
            else:
                self.knowledge_file = path
        else:
            self.knowledge_file = None
            for path in self.config.knowledge_base_paths:
                if path.exists():
                    self.knowledge_file = path
                    break

    def load_concepts(self) -> list[PsychologyConcept]:
        """Load all psychology concepts"""
        if not self.knowledge_file or not self.knowledge_file.exists():
            logger.warning("Psychology knowledge base not found")
            logger.info("Searched locations:")
            logger.info("  - ai/pixel/knowledge/psychology_knowledge_base_optimized.json")
            logger.info(
                "  - ai/training_data_consolidated/psychology_knowledge_base_optimized.json"
            )
            return []

        try:
            with open(self.knowledge_file) as f:
                data = json.load(f)

            concepts = []

            # Handle different possible formats
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                # Get concepts - could be a list or a dict
                concepts_data = data.get("concepts", data.get("knowledge", []))
                if isinstance(concepts_data, dict):
                    # Convert dict to list of values
                    items = list(concepts_data.values())
                elif isinstance(concepts_data, list):
                    items = concepts_data
                else:
                    logger.error(
                        f"Unexpected concepts format in {self.knowledge_file}: {type(concepts_data)}"
                    )
                    return []
            else:
                logger.error(f"Unexpected data format in {self.knowledge_file}: {type(data)}")
                return []

            for idx, item in enumerate(items):
                try:
                    # Ensure item is a dict
                    if not isinstance(item, dict):
                        logger.warning(f"Skipping non-dict item at index {idx}: {type(item)}")
                        continue

                    concept = PsychologyConcept(
                        concept_id=item.get("id", item.get("concept_id", f"psych_{idx:04d}")),
                        title=item.get("title", item.get("name", item.get("concept", ""))),
                        content=item.get(
                            "content", item.get("description", item.get("definition", ""))
                        ),
                        category=item.get("category", item.get("type", "general")),
                        subcategory=item.get("subcategory", item.get("subtype")),
                        related_concepts=item.get("related_concepts", item.get("related", [])),
                        therapeutic_approaches=item.get(
                            "therapeutic_approaches", item.get("approaches", [])
                        ),
                        source=item.get("source", "psychology_knowledge"),
                    )

                    # Only add if has meaningful content
                    if concept.title and concept.content:
                        concepts.append(concept)

                except Exception as e:
                    logger.error(f"Error parsing concept {idx}: {e}")
                    continue

            logger.info(f"Loaded {len(concepts)} psychology concepts from {self.knowledge_file}")
            return concepts

        except Exception as e:
            logger.error(f"Failed to load psychology knowledge: {e}")
            return []

    def load_by_category(self, category: str) -> list[PsychologyConcept]:
        """Load concepts filtered by category"""
        all_concepts = self.load_concepts()
        filtered = [c for c in all_concepts if c.category == category]
        logger.info(f"Loaded {len(filtered)} concepts for category '{category}'")
        return filtered

    def get_statistics(self) -> dict:
        """Get statistics about loaded psychology knowledge"""
        concepts = self.load_concepts()

        if not concepts:
            return {
                "total_concepts": 0,
                "categories": {},
                "subcategories": {},
                "therapeutic_approaches": {},
            }

        # Count by category
        categories = {}
        for concept in concepts:
            categories[concept.category] = categories.get(concept.category, 0) + 1

        # Count by subcategory
        subcategories = {}
        for concept in concepts:
            if concept.subcategory:
                subcategories[concept.subcategory] = subcategories.get(concept.subcategory, 0) + 1

        # Count therapeutic approaches
        approaches = {}
        for concept in concepts:
            for approach in concept.therapeutic_approaches:
                approaches[approach] = approaches.get(approach, 0) + 1

        return {
            "total_concepts": len(concepts),
            "categories": categories,
            "subcategories": subcategories,
            "therapeutic_approaches": approaches,
            "file_path": str(self.knowledge_file) if self.knowledge_file else None,
        }

    def convert_to_training_format(
        self, concepts: list[PsychologyConcept] | None = None
    ) -> list[dict]:
        """Convert psychology concepts to standard training format"""
        if concepts is None:
            concepts = self.load_concepts()

        training_data = [concept.to_training_format() for concept in concepts]
        logger.info(f"Converted {len(training_data)} psychology concepts to training format")
        return training_data

    def check_knowledge_base_exists(self) -> bool:
        """Check if psychology knowledge base exists"""
        return self.knowledge_file is not None and self.knowledge_file.exists()

    def get_setup_instructions(self) -> str:
        """Get instructions for setting up psychology knowledge base"""
        return """
To set up the psychology knowledge base:

1. The knowledge base should contain 4,867+ psychology concepts

2. Expected location (one of):
   - ai/pixel/knowledge/psychology_knowledge_base_optimized.json
   - ai/training_data_consolidated/psychology_knowledge_base_optimized.json

3. Expected format:
   {
     "concepts": [
       {
         "id": "dsm5_ptsd",
         "title": "Post-Traumatic Stress Disorder (PTSD)",
         "content": "PTSD is a mental health condition...",
         "category": "clinical_diagnosis",
         "subcategory": "trauma_disorders",
         "related_concepts": ["trauma", "flashbacks", "hypervigilance"],
         "therapeutic_approaches": ["EMDR", "CPT", "PE"],
         "source": "DSM-5"
       },
       ...
     ]
   }

4. Sources to include:
   - DSM-5 clinical definitions
   - Therapeutic techniques (CBT, DBT, ACT, EMDR, etc.)
   - Psychology concepts and theories
   - Mental health knowledge
   - Expert transcripts (Tim Fletcher, etc.)

5. If you have the data in a different format, convert it to match the expected structure
"""


def load_psychology_knowledge(knowledge_base_path: str | None = None) -> list[dict]:
    """
    Convenience function to load psychology knowledge

    Args:
        knowledge_base_path: Optional path to knowledge base file

    Returns:
        List of training examples in standard format
    """
    loader = PsychologyKnowledgeLoader(knowledge_base_path)

    if not loader.check_knowledge_base_exists():
        logger.warning("Psychology knowledge base not found!")
        logger.info(loader.get_setup_instructions())
        return []

    return loader.convert_to_training_format()


if __name__ == "__main__":
    # Test the loader
    loader = PsychologyKnowledgeLoader()

    logger.info("Psychology Knowledge Base Loader")
    logger.info("=" * 60)

    if not loader.check_knowledge_base_exists():
        logger.warning("\n❌ Psychology knowledge base not found!")
        logger.info(loader.get_setup_instructions())
    else:
        logger.info(f"\n✅ Psychology knowledge base found: {loader.knowledge_file}")

        # Load and show statistics
        stats = loader.get_statistics()
        logger.info("\n📊 Statistics:")
        logger.info(f"   Total concepts: {stats['total_concepts']}")
        logger.info(f"   Categories: {len(stats['categories'])}")

        logger.info("\n📚 Top Categories:")
        for category, count in sorted(
            stats["categories"].items(), key=lambda x: x[1], reverse=True
        )[:10]:
            logger.info(f"   {category}: {count}")

        if stats["therapeutic_approaches"]:
            logger.info("\n🔧 Therapeutic Approaches:")
            for approach, count in sorted(
                stats["therapeutic_approaches"].items(), key=lambda x: x[1], reverse=True
            )[:10]:
                logger.info(f"   {approach}: {count}")

        # Load training data
        training_data = loader.convert_to_training_format()
        logger.info(f"\n✅ Loaded {len(training_data)} training examples")

        if training_data:
            logger.info("\n📝 Sample example:")
            sample = training_data[0]
            logger.info(f"   Category: {sample['metadata']['category']}")
            logger.info(f"   Concept: {sample['metadata']['concept_id']}")
            logger.info(f"   Text: {sample['text'][:200]}...")
