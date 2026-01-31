#!/usr/bin/env python3
"""
Test script for Tier6KnowledgeLoader with enhanced psychology knowledge base.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
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

# Add the tier loaders path
tier_loaders_path = project_root / "ai" / "dataset_pipeline" / "ingestion" / "tier_loaders"
sys.path.insert(0, str(tier_loaders_path))

from tier6_knowledge_loader import Tier6KnowledgeLoader

def test_tier6_loader():
    """Test the Tier6KnowledgeLoader with enhanced psychology knowledge base."""
    print("Testing Tier6KnowledgeLoader with enhanced psychology knowledge base...")

    # Initialize the loader
    loader = Tier6KnowledgeLoader()

    # Load datasets
    datasets = loader.load_datasets()

    print(f"Loaded {len(datasets)} datasets:")
    for name, conversations in datasets.items():
        print(f"  {name}: {len(conversations)} conversations")

        # Show first conversation as example
        if conversations:
            conv = conversations[0]
            print(f"    Example conversation ID: {conv.conversation_id}")
            print(f"    Source: {conv.source}")
            print(f"    Messages: {len(conv.messages)}")
            if conv.messages:
                print(f"    First message role: {conv.messages[0].role}")
                print(f"    First message content: {conv.messages[0].content[:100]}...")
            print()

if __name__ == "__main__":
    test_tier6_loader()