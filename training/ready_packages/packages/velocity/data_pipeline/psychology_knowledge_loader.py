#!/usr/bin/env python3
"""
Psychology Knowledge Base Loader for Training Pipeline Integration
Loads 4,867+ psychology concepts, DSM-5 definitions, and therapeutic techniques
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.psychology_knowledge_loader")


@dataclass
class PsychologyConcept:
    """Structured psychology knowledge concept"""
    concept_id: str
    title: str
    content: str
    category: str
    subcategory: Optional[str] = None
    related_concepts: List[str] = None
    therapeutic_approaches: List[str] = None
    source: str = "psychology_knowledge"
    
    def __post_init__(self):
        if self.related_concepts is None:
            self.related_concepts = []
        if self.therapeutic_approaches is None:
            self.therapeutic_approaches = []
    
    def to_training_format(self) -> Dict:
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
                "is_edge_case": False
            }
        }


class PsychologyKnowledgeLoader:
    """Loader for psychology knowledge base"""
    
    def __init__(self, knowledge_base_path: Optional[str] = None):
        if knowledge_base_path:
            self.knowledge_file = Path(knowledge_base_path)
        else:
            # Try multiple possible locations
            possible_paths = [
                Path("ai/pixel/knowledge/psychology_knowledge_base_optimized.json"),
                Path("ai/training_data_consolidated/psychology_knowledge_base_optimized.json"),
                Path("ai/dataset_pipeline/data/psychology_knowledge_base.json"),
            ]
            self.knowledge_file = None
            for path in possible_paths:
                if path.exists():
                    self.knowledge_file = path
                    break
    
    def load_concepts(self) -> List[PsychologyConcept]:
        """Load all psychology concepts"""
        if not self.knowledge_file or not self.knowledge_file.exists():
            logger.warning(f"Psychology knowledge base not found")
            logger.info("Searched locations:")
            logger.info("  - ai/pixel/knowledge/psychology_knowledge_base_optimized.json")
            logger.info("  - ai/training_data_consolidated/psychology_knowledge_base_optimized.json")
            return []
        
        try:
            with open(self.knowledge_file, 'r') as f:
                data = json.load(f)
            
            concepts = []
            
            # Handle different possible formats
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get('concepts', data.get('knowledge', []))
            else:
                logger.error(f"Unexpected data format in {self.knowledge_file}")
                return []
            
            for idx, item in enumerate(items):
                try:
                    concept = PsychologyConcept(
                        concept_id=item.get('id', item.get('concept_id', f"psych_{idx:04d}")),
                        title=item.get('title', item.get('name', item.get('concept', ''))),
                        content=item.get('content', item.get('description', item.get('definition', ''))),
                        category=item.get('category', item.get('type', 'general')),
                        subcategory=item.get('subcategory', item.get('subtype')),
                        related_concepts=item.get('related_concepts', item.get('related', [])),
                        therapeutic_approaches=item.get('therapeutic_approaches', item.get('approaches', [])),
                        source=item.get('source', 'psychology_knowledge')
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
    
    def load_by_category(self, category: str) -> List[PsychologyConcept]:
        """Load concepts filtered by category"""
        all_concepts = self.load_concepts()
        filtered = [c for c in all_concepts if c.category == category]
        logger.info(f"Loaded {len(filtered)} concepts for category '{category}'")
        return filtered
    
    def get_statistics(self) -> Dict:
        """Get statistics about loaded psychology knowledge"""
        concepts = self.load_concepts()
        
        if not concepts:
            return {
                "total_concepts": 0,
                "categories": {},
                "subcategories": {},
                "therapeutic_approaches": {}
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
            "file_path": str(self.knowledge_file) if self.knowledge_file else None
        }
    
    def convert_to_training_format(self, concepts: Optional[List[PsychologyConcept]] = None) -> List[Dict]:
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


def load_psychology_knowledge(knowledge_base_path: Optional[str] = None) -> List[Dict]:
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
    
    print("Psychology Knowledge Base Loader")
    print("=" * 60)
    
    if not loader.check_knowledge_base_exists():
        print("\n❌ Psychology knowledge base not found!")
        print(loader.get_setup_instructions())
    else:
        print(f"\n✅ Psychology knowledge base found: {loader.knowledge_file}")
        
        # Load and show statistics
        stats = loader.get_statistics()
        print(f"\n📊 Statistics:")
        print(f"   Total concepts: {stats['total_concepts']}")
        print(f"   Categories: {len(stats['categories'])}")
        
        print(f"\n📚 Top Categories:")
        for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {category}: {count}")
        
        if stats['therapeutic_approaches']:
            print(f"\n🔧 Therapeutic Approaches:")
            for approach, count in sorted(stats['therapeutic_approaches'].items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   {approach}: {count}")
        
        # Load training data
        training_data = loader.convert_to_training_format()
        print(f"\n✅ Loaded {len(training_data)} training examples")
        
        if training_data:
            print(f"\n📝 Sample example:")
            sample = training_data[0]
            print(f"   Category: {sample['metadata']['category']}")
            print(f"   Concept: {sample['metadata']['concept_id']}")
            print(f"   Text: {sample['text'][:200]}...")
