#!/usr/bin/env python3
"""
Simple test script to verify book references in enhanced psychology knowledge base.
"""

import json
from pathlib import Path

def check_book_references():
    """Check for book references in the enhanced psychology knowledge base."""
    kb_path = Path("ai/pixel/knowledge/enhanced_psychology_knowledge_base.json")

    if not kb_path.exists():
        print(f"Knowledge base not found: {kb_path}")
        return

    with open(kb_path, "r", encoding="utf-8") as f:
        knowledge_base = json.load(f)

    concepts = knowledge_base.get("concepts", {})
    print(f"Total concepts in knowledge base: {len(concepts)}")

    # Count book references
    book_refs = []
    for concept_id, concept in concepts.items():
        if concept.get("category") == "psychology_book_reference":
            book_refs.append((concept_id, concept))

    print(f"Book reference concepts: {len(book_refs)}")

    # Show examples
    for i, (concept_id, concept) in enumerate(book_refs[:3]):
        print(f"\nBook reference {i+1}:")
        print(f"  Concept ID: {concept_id}")
        print(f"  Name: {concept.get('name', 'N/A')}")
        print(f"  Definition preview: {concept.get('definition', '')[:100]}...")

if __name__ == "__main__":
    check_book_references()