"""
Clinical Knowledge Embedder

Creates vector embeddings for all psychology knowledge items to enable
efficient similarity search and retrieval during training.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime
import hashlib

try:
    import numpy as np
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    import faiss
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    logging.warning(
        "Dependencies not available. Install: uv pip install faiss-cpu sentence-transformers numpy pandas")

from .psychology_loader import PsychologyKnowledgeLoader, PsychologyKnowledge
from .therapeutic_conversation_schema import TherapeuticConversation


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    model_name: str = "all-MiniLM-L6-v2"  # Fast, good quality embeddings
    batch_size: int = 32
    max_length: int = 512
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    embedding_dimension: int = 384  # Dimension for all-MiniLM-L6-v2


@dataclass
class KnowledgeItem:
    """Represents a knowledge item with its embedding."""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    knowledge_type: str = "general"
    source: str = "unknown"
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        """Generate content hash for caching."""
        if not hasattr(self, 'content_hash'):
            self.content_hash = hashlib.md5(self.content.encode()).hexdigest()


class ClinicalKnowledgeEmbedder:
    """Creates and manages vector embeddings for clinical psychology knowledge."""

    def __init__(
            self,
            config: Optional[EmbeddingConfig] = None,
            project_root: Optional[Path] = None):
        """Initialize the embedder with configuration."""
        self.config = config or EmbeddingConfig()
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.psychology_loader = PsychologyKnowledgeLoader(project_root)

        # Initialize embedding model (when dependencies are available)
        self.embedding_model = None
        self.embeddings_cache_path = self.project_root / \
            "ai" / "pixel" / "data" / "embeddings_cache.pkl"
        self.embeddings_cache = {}

        # Knowledge storage
        self.knowledge_items: List[KnowledgeItem] = []
        self.embeddings_matrix: Optional[np.ndarray] = None

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        if DEPENDENCIES_AVAILABLE:
            self._initialize_embedding_model()
            self._load_embeddings_cache()
        else:
            self.logger.warning(
                "Dependencies not available. Embedder will work in mock mode.")

    def _initialize_embedding_model(self):
        """Initialize the sentence transformer model."""
        try:
            self.embedding_model = SentenceTransformer(self.config.model_name)
            self.logger.info(
                f"Initialized embedding model: {self.config.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {e}")
            self.embedding_model = None

    def _load_embeddings_cache(self):
        """Load cached embeddings if available."""
        if self.embeddings_cache_path.exists():
            try:
                with open(self.embeddings_cache_path, 'rb') as f:
                    self.embeddings_cache = pickle.load(f)
                self.logger.info(
                    f"Loaded {len(self.embeddings_cache)} cached embeddings")
            except Exception as e:
                self.logger.error(f"Failed to load embeddings cache: {e}")
                self.embeddings_cache = {}

    def _save_embeddings_cache(self):
        """Save embeddings cache to disk."""
        if not self.config.cache_embeddings:
            return

        try:
            self.embeddings_cache_path.parent.mkdir(
                parents=True, exist_ok=True)
            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(self.embeddings_cache, f)
            self.logger.info(
                f"Saved {len(self.embeddings_cache)} embeddings to cache")
        except Exception as e:
            self.logger.error(f"Failed to save embeddings cache: {e}")

    def extract_psychology_knowledge_items(self) -> List[KnowledgeItem]:
        """Extract all psychology knowledge items from available sources."""
        knowledge_items = []

        try:
            # Load DSM-5 knowledge
            dsm5_knowledge = self.psychology_loader.load_dsm5_knowledge()
            for item in dsm5_knowledge:
                knowledge_items.append(KnowledgeItem(
                    id=f"dsm5_{item.id}",
                    content=f"{item.title}\n\n{item.content}",
                    metadata={
                        "source": "dsm5",
                        "title": item.title,
                        "knowledge_type": item.knowledge_type,
                        "original_metadata": item.metadata
                    },
                    knowledge_type="dsm5",
                    source="dsm5_knowledge_base"
                ))

            # Load PDM-2 knowledge
            pdm2_knowledge = self.psychology_loader.load_pdm2_knowledge()
            for item in pdm2_knowledge:
                knowledge_items.append(KnowledgeItem(
                    id=f"pdm2_{item.id}",
                    content=f"{item.title}\n\n{item.content}",
                    metadata={
                        "source": "pdm2",
                        "title": item.title,
                        "knowledge_type": item.knowledge_type,
                        "original_metadata": item.metadata
                    },
                    knowledge_type="pdm2",
                    source="pdm2_knowledge_base"
                ))

            # Load therapeutic conversations
            conversations = self.psychology_loader.load_therapeutic_conversations()
            for conv in conversations:
                # Create knowledge items from conversation content
                conversation_content = self._extract_conversation_content(conv)
                knowledge_items.append(
                    KnowledgeItem(
                        id=f"conversation_{conv.id}",
                        content=conversation_content,
                        metadata={
                            "source": "therapeutic_conversation",
                            "modality": conv.modality.value if conv.modality else "unknown",
                            "clinical_context": conv.clinical_context.__dict__ if conv.clinical_context else {},
                            "turn_count": len(
                                conv.turns)},
                        knowledge_type="therapeutic_conversation",
                        source="conversation_dataset"))

            # Load clinical datasets
            clinical_data = self.psychology_loader.load_clinical_datasets()
            for item in clinical_data:
                knowledge_items.append(KnowledgeItem(
                    id=f"clinical_{item.id}",
                    content=f"{item.title}\n\n{item.content}",
                    metadata={
                        "source": "clinical_dataset",
                        "title": item.title,
                        "knowledge_type": item.knowledge_type,
                        "original_metadata": item.metadata
                    },
                    knowledge_type="clinical",
                    source="clinical_datasets"
                ))

        except Exception as e:
            self.logger.error(f"Error extracting psychology knowledge: {e}")
            # Create mock data for testing when dependencies aren't available
            knowledge_items = self._create_mock_knowledge_items()

        self.logger.info(f"Extracted {len(knowledge_items)} knowledge items")
        return knowledge_items

    def _extract_conversation_content(
            self, conversation: TherapeuticConversation) -> str:
        """Extract meaningful content from therapeutic conversation."""
        content_parts = []

        if conversation.title:
            content_parts.append(f"Title: {conversation.title}")

        if conversation.clinical_context:
            if conversation.clinical_context.dsm5_categories:
                content_parts.append(
                    f"DSM-5 Categories: {', '.join(conversation.clinical_context.dsm5_categories)}")
            if conversation.clinical_context.presenting_concerns:
                content_parts.append(
                    f"Presenting Concerns: {', '.join(conversation.clinical_context.presenting_concerns)}")

        # Extract key therapeutic exchanges
        for i, turn in enumerate(conversation.turns):
            role = "Therapist" if turn.role.value == "therapist" else "Client"
            content_parts.append(f"{role}: {turn.content}")

            if turn.clinical_rationale:
                content_parts.append(
                    f"Clinical Rationale: {turn.clinical_rationale}")

            # Limit to prevent overly long content
            if i >= 10:  # First 10 turns
                break

        return "\n\n".join(content_parts)

    def _create_mock_knowledge_items(self) -> List[KnowledgeItem]:
        """Create mock knowledge items for testing when dependencies aren't available."""
        mock_items = [
            KnowledgeItem(
                id="mock_dsm5_depression",
                content="Major Depressive Disorder: Characterized by persistent sadness, loss of interest, and impaired functioning for at least two weeks.",
                metadata={"source": "mock_dsm5", "category": "mood_disorders"},
                knowledge_type="dsm5",
                source="mock_data"
            ),
            KnowledgeItem(
                id="mock_cbt_technique",
                content="Cognitive Behavioral Therapy: Focuses on identifying and changing negative thought patterns and behaviors.",
                metadata={"source": "mock_therapy", "modality": "cbt"},
                knowledge_type="therapeutic_technique",
                source="mock_data"
            ),
            KnowledgeItem(
                id="mock_attachment_theory",
                content="Attachment Theory: Describes the dynamics of long-term relationships between humans, particularly in early childhood development.",
                metadata={"source": "mock_pdm2", "category": "attachment"},
                knowledge_type="pdm2",
                source="mock_data"
            )
        ]
        return mock_items

    def generate_embeddings(
            self, knowledge_items: Optional[List[KnowledgeItem]] = None) -> List[KnowledgeItem]:
        """Generate embeddings for knowledge items."""
        if knowledge_items is None:
            knowledge_items = self.knowledge_items

        if not DEPENDENCIES_AVAILABLE or self.embedding_model is None:
            self.logger.warning(
                "Generating mock embeddings (dependencies not available)")
            return self._generate_mock_embeddings(knowledge_items)

        items_to_embed = []
        texts_to_embed = []

        # Check cache and prepare items for embedding
        for item in knowledge_items:
            if self.config.cache_embeddings and item.content_hash in self.embeddings_cache:
                item.embedding = self.embeddings_cache[item.content_hash]
                self.logger.debug(f"Using cached embedding for {item.id}")
            else:
                items_to_embed.append(item)
                texts_to_embed.append(item.content[:self.config.max_length])

        # Generate embeddings for uncached items
        if texts_to_embed:
            self.logger.info(
                f"Generating embeddings for {len(texts_to_embed)} items")

            try:
                embeddings = self.embedding_model.encode(
                    texts_to_embed,
                    batch_size=self.config.batch_size,
                    normalize_embeddings=self.config.normalize_embeddings,
                    show_progress_bar=True
                )

                # Assign embeddings and update cache
                for item, embedding in zip(items_to_embed, embeddings):
                    item.embedding = embedding
                    if self.config.cache_embeddings:
                        self.embeddings_cache[item.content_hash] = embedding

                # Save updated cache
                if self.config.cache_embeddings:
                    self._save_embeddings_cache()

            except Exception as e:
                self.logger.error(f"Failed to generate embeddings: {e}")
                return self._generate_mock_embeddings(knowledge_items)

        self.logger.info(
            f"Generated embeddings for {len(knowledge_items)} knowledge items")
        return knowledge_items

    def _generate_mock_embeddings(
            self, knowledge_items: List[KnowledgeItem]) -> List[KnowledgeItem]:
        """Generate mock embeddings for testing."""
        if not DEPENDENCIES_AVAILABLE:
            # Create mock numpy arrays
            for item in knowledge_items:
                # Create deterministic mock embedding based on content hash
                hash_int = int(item.content_hash[:8], 16)
                mock_embedding = [
                    (hash_int +
                     i) %
                    1000 /
                    1000.0 for i in range(
                        self.config.embedding_dimension)]
                item.embedding = mock_embedding
        else:
            # Use numpy for proper mock embeddings
            for item in knowledge_items:
                hash_int = int(item.content_hash[:8], 16)
                np.random.seed(hash_int)
                item.embedding = np.random.normal(
                    0, 1, self.config.embedding_dimension).astype(
                    np.float32)

        return knowledge_items

    def create_embeddings_matrix(
            self, knowledge_items: Optional[List[KnowledgeItem]] = None) -> np.ndarray:
        """Create embeddings matrix from knowledge items."""
        if knowledge_items is None:
            knowledge_items = self.knowledge_items

        if not knowledge_items:
            raise ValueError("No knowledge items provided")

        # Ensure all items have embeddings
        items_with_embeddings = [
            item for item in knowledge_items if item.embedding is not None]

        if not items_with_embeddings:
            raise ValueError("No knowledge items have embeddings")

        if DEPENDENCIES_AVAILABLE:
            embeddings_matrix = np.vstack(
                [item.embedding for item in items_with_embeddings])
        else:
            # Mock matrix creation
            embeddings_matrix = [
                item.embedding for item in items_with_embeddings]

        self.embeddings_matrix = embeddings_matrix
        self.knowledge_items = items_with_embeddings

        self.logger.info(
            f"Created embeddings matrix: {len(items_with_embeddings)} items x {self.config.embedding_dimension} dimensions")
        return embeddings_matrix

    def save_embeddings(self, output_path: Optional[Path] = None) -> Path:
        """Save embeddings and knowledge items to disk."""
        if output_path is None:
            output_path = self.project_root / "ai" / "pixel" / \
                "data" / "clinical_knowledge_embeddings.pkl"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        embeddings_data = {
            "knowledge_items": self.knowledge_items,
            "embeddings_matrix": self.embeddings_matrix,
            "config": self.config,
            "created_at": datetime.now(),
            "item_count": len(self.knowledge_items),
            "embedding_dimension": self.config.embedding_dimension
        }

        try:
            with open(output_path, 'wb') as f:
                pickle.dump(embeddings_data, f)

            self.logger.info(f"Saved embeddings to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Failed to save embeddings: {e}")
            raise

    def load_embeddings(self, input_path: Optional[Path] = None) -> bool:
        """Load embeddings and knowledge items from disk."""
        if input_path is None:
            input_path = self.project_root / "ai" / "pixel" / \
                "data" / "clinical_knowledge_embeddings.pkl"

        if not input_path.exists():
            self.logger.warning(f"Embeddings file not found: {input_path}")
            return False

        try:
            with open(input_path, 'rb') as f:
                embeddings_data = pickle.load(f)

            self.knowledge_items = embeddings_data["knowledge_items"]
            self.embeddings_matrix = embeddings_data["embeddings_matrix"]

            self.logger.info(
                f"Loaded {len(self.knowledge_items)} knowledge items with embeddings")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load embeddings: {e}")
            return False

    def process_all_knowledge(self) -> Tuple[List[KnowledgeItem], np.ndarray]:
        """Complete pipeline: extract knowledge, generate embeddings, create matrix."""
        self.logger.info("Starting complete knowledge processing pipeline")

        # Extract knowledge items
        knowledge_items = self.extract_psychology_knowledge_items()

        # Generate embeddings
        knowledge_items = self.generate_embeddings(knowledge_items)

        # Create embeddings matrix
        embeddings_matrix = self.create_embeddings_matrix(knowledge_items)

        # Save results
        self.save_embeddings()

        self.logger.info(
            "Knowledge processing pipeline completed successfully")
        return knowledge_items, embeddings_matrix

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about the embeddings."""
        if not self.knowledge_items:
            return {"error": "No knowledge items loaded"}

        stats = {
            "total_items": len(
                self.knowledge_items),
            "embedding_dimension": self.config.embedding_dimension,
            "knowledge_types": {},
            "sources": {},
            "items_with_embeddings": sum(
                1 for item in self.knowledge_items if item.embedding is not None),
            "cache_size": len(
                self.embeddings_cache) if hasattr(
                    self,
                'embeddings_cache') else 0}

        # Count by knowledge type and source
        for item in self.knowledge_items:
            stats["knowledge_types"][item.knowledge_type] = stats["knowledge_types"].get(
                item.knowledge_type, 0) + 1
            stats["sources"][item.source] = stats["sources"].get(
                item.source, 0) + 1

        return stats


def main():
    """Test the clinical knowledge embedder."""
    print("Testing Clinical Knowledge Embedder")

    # Initialize embedder
    config = EmbeddingConfig(
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
        cache_embeddings=True
    )

    embedder = ClinicalKnowledgeEmbedder(config)

    # Process all knowledge
    try:
        knowledge_items, embeddings_matrix = embedder.process_all_knowledge()

        # Print statistics
        stats = embedder.get_embedding_stats()
        print("\nEmbedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        print(
            f"\nSuccessfully processed {len(knowledge_items)} knowledge items")
        print(
            f"Embeddings matrix shape: {embeddings_matrix.shape if hasattr(embeddings_matrix, 'shape') else len(embeddings_matrix)}")

    except Exception as e:
        print(f"Error during processing: {e}")


if __name__ == "__main__":
    main()
