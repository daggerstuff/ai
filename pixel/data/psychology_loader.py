"""
Psychology Knowledge Base Loader for Dataset Pipeline

This module provides functionality to load and process psychology knowledge from various sources
including FAISS indexes, clinical datasets, personality assessments, and therapy content.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import faiss
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Vector search features will be disabled.")


@dataclass
class PsychologyKnowledge:
    """Represents a piece of psychology knowledge."""
    id: str
    title: str
    content: str
    knowledge_type: str  # dsm, pdm, personality, attachment, clinical, therapy
    source: str
    metadata: Dict[str, Any]

    @property
    def content_length(self) -> int:
        """Return the character length of the content."""
        return len(self.content)

    @property
    def has_clinical_terms(self) -> bool:
        """Check if content contains clinical terminology."""
        clinical_terms = [
            'disorder', 'diagnosis', 'symptom', 'therapy', 'treatment',
            'patient', 'client', 'assessment', 'intervention', 'pathology'
        ]
        content_lower = self.content.lower()
        return any(term in content_lower for term in clinical_terms)


class PsychologyKnowledgeLoader:
    """Loads psychology knowledge from various sources in the project."""

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the loader with project paths."""
        if project_root is None:
            # Try to find project root from current location
            current = Path.cwd()
            while current.parent != current:
                if (current / "ai").exists():
                    project_root = current
                    break
                current = current.parent

            if project_root is None:
                project_root = Path.cwd()

        self.project_root = Path(project_root)
        self.ai_root = self.project_root / "ai"

        # Define knowledge source paths
        self.psychology_test_dir = self.ai_root / "1.PsychologyTest"
        self.faiss_index_dir = self.psychology_test_dir / \
            "knowledge" / "faiss_index_all_documents"
        self.datasets_dir = self.ai_root / "datasets"
        self.books_dir = self.ai_root / "Books"
        self.mental_arena_dir = self.ai_root / \
            "6-2-leftovers" / "Datasets" / "MentalArena"

        print(
            f"Initialized PsychologyKnowledgeLoader with root: {self.project_root}")

    def get_available_knowledge_sources(self) -> List[Dict[str, Any]]:
        """Get information about available psychology knowledge sources."""
        sources = []

        # Check FAISS index
        if self.faiss_index_dir.exists() and FAISS_AVAILABLE:
            faiss_files = list(self.faiss_index_dir.glob(
                "*.faiss")) + list(self.faiss_index_dir.glob("*.pkl"))
            if faiss_files:
                sources.append({
                    "name": "FAISS Psychology Knowledge Index",
                    "path": self.faiss_index_dir,
                    "type": "faiss_vector_index",
                    "description": "Vectorized psychology documents (DSM-5, PDM-2, etc.)",
                    "files": [f.name for f in faiss_files]
                })

        # Check clinical knowledge datasets
        if self.mental_arena_dir.exists():
            clinical_files = list(
                self.mental_arena_dir.rglob("*clinical*.jsonl"))
            for file in clinical_files:
                sources.append({
                    "name": f"Clinical Knowledge - {file.stem}",
                    "path": file,
                    "type": "clinical_dataset",
                    "description": "Clinical knowledge test questions and answers",
                    "size": file.stat().st_size if file.exists() else 0
                })

        # Check personality datasets
        personality_files = list(self.datasets_dir.rglob(
            "*personality*.csv")) if self.datasets_dir.exists() else []
        for file in personality_files:
            sources.append({
                "name": f"Personality Data - {file.stem}",
                "path": file,
                "type": "personality_dataset",
                "description": "Personality assessment and trait data",
                "size": file.stat().st_size
            })

        # Check therapy books
        if self.books_dir.exists():
            therapy_books = list(self.books_dir.rglob("*.pdf"))
            for book in therapy_books:
                if any(term in book.name.lower()
                       for term in ['therapy', 'clinical', 'psychology', 'dbt']):
                    sources.append({
                        "name": f"Therapy Book - {book.stem}",
                        "path": book,
                        "type": "therapy_book",
                        "description": "Clinical therapy and treatment guide",
                        "size": book.stat().st_size
                    })

        return sources

    def load_faiss_knowledge(
            self,
            query: str = "",
            top_k: int = 10) -> List[PsychologyKnowledge]:
        """Load knowledge from FAISS vector index."""
        if not FAISS_AVAILABLE:
            print("FAISS not available. Cannot load vector knowledge.")
            return []

        if not self.faiss_index_dir.exists():
            print(f"FAISS index not found at: {self.faiss_index_dir}")
            return []

        try:
            # Load the FAISS index (requires OpenAI API key for embeddings)
            embedding_model = OpenAIEmbeddings()
            vectorstore = FAISS.load_local(
                str(self.faiss_index_dir),
                embedding_model,
                allow_dangerous_deserialization=True
            )

            if query:
                # Search for relevant documents
                docs = vectorstore.similarity_search(query, k=top_k)
            else:
                # Get sample documents
                docs = vectorstore.similarity_search(
                    "psychology clinical assessment", k=top_k)

            knowledge_items = []
            for i, doc in enumerate(docs):
                knowledge = PsychologyKnowledge(
                    id=f"faiss_{i}",
                    title=f"Psychology Document {i+1}",
                    content=doc.page_content,
                    knowledge_type="faiss_indexed",
                    source="FAISS Vector Index",
                    metadata=doc.metadata if hasattr(doc, 'metadata') else {}
                )
                knowledge_items.append(knowledge)

            return knowledge_items

        except Exception as e:
            print(f"Error loading FAISS knowledge: {e}")
            return []

    def load_clinical_knowledge(
            self,
            limit: Optional[int] = None) -> List[PsychologyKnowledge]:
        """Load clinical knowledge from test datasets."""
        knowledge_items = []

        if not self.mental_arena_dir.exists():
            print("Mental Arena directory not found")
            return knowledge_items

        clinical_files = list(self.mental_arena_dir.rglob("*clinical*.jsonl"))

        for file_path in clinical_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if limit and len(knowledge_items) >= limit:
                            break

                        try:
                            data = json.loads(line.strip())

                            # Extract question and answer
                            question = data.get('question', '')
                            choices = data.get('choices', [])
                            answer = data.get('answer', '')
                            subject = data.get('subject', 'clinical_knowledge')

                            # Combine into content
                            content = f"Question: {question}\n"
                            if choices:
                                content += "Choices:\n" + \
                                    "\n".join([f"  {i}: {choice}" for i, choice in enumerate(choices)])
                                content += f"\nCorrect Answer: {answer}"

                            knowledge = PsychologyKnowledge(
                                id=f"clinical_{file_path.stem}_{line_num}",
                                title=f"Clinical Knowledge Question {line_num + 1}",
                                content=content,
                                knowledge_type="clinical",
                                source=file_path.name,
                                metadata={
                                    "subject": subject,
                                    "file": str(file_path),
                                    "line_number": line_num})
                            knowledge_items.append(knowledge)

                        except json.JSONDecodeError as e:
                            print(
                                f"Error parsing line {line_num} in {file_path}: {e}")
                            continue

            except Exception as e:
                print(f"Error reading clinical file {file_path}: {e}")
                continue

        return knowledge_items

    def load_personality_knowledge(
            self, limit: Optional[int] = None) -> List[PsychologyKnowledge]:
        """Load personality assessment data."""
        knowledge_items = []

        if not self.datasets_dir.exists():
            return knowledge_items

        personality_files = list(self.datasets_dir.rglob("*personality*.csv"))

        for file_path in personality_files:
            try:
                df = pd.read_csv(file_path)

                for index, row in df.iterrows():
                    if limit and len(knowledge_items) >= limit:
                        break

                    # Convert row to content
                    content_parts = []
                    for col, value in row.items():
                        if pd.notna(value):
                            content_parts.append(f"{col}: {value}")

                    content = "\n".join(content_parts)

                    knowledge = PsychologyKnowledge(
                        id=f"personality_{file_path.stem}_{index}",
                        title=f"Personality Data Entry {index + 1}",
                        content=content,
                        knowledge_type="personality",
                        source=file_path.name,
                        metadata={
                            "file": str(file_path),
                            "row_index": index,
                            "columns": list(row.index)
                        }
                    )
                    knowledge_items.append(knowledge)

            except Exception as e:
                print(f"Error reading personality file {file_path}: {e}")
                continue

        return knowledge_items

    def load_all_knowledge(
            self,
            limit_per_type: int = 50) -> List[PsychologyKnowledge]:
        """Load knowledge from all available sources."""
        all_knowledge = []

        print("Loading clinical knowledge...")
        clinical = self.load_clinical_knowledge(limit=limit_per_type)
        all_knowledge.extend(clinical)
        print(f"Loaded {len(clinical)} clinical knowledge items")

        print("Loading personality knowledge...")
        personality = self.load_personality_knowledge(limit=limit_per_type)
        all_knowledge.extend(personality)
        print(f"Loaded {len(personality)} personality knowledge items")

        try:
            print("Loading FAISS knowledge...")
            faiss_knowledge = self.load_faiss_knowledge(top_k=limit_per_type)
            all_knowledge.extend(faiss_knowledge)
            print(f"Loaded {len(faiss_knowledge)} FAISS knowledge items")
        except Exception as e:
            print(f"Could not load FAISS knowledge: {e}")

        return all_knowledge

    def search_knowledge(
            self,
            query: str,
            knowledge_type: Optional[str] = None,
            limit: int = 10) -> List[PsychologyKnowledge]:
        """Search knowledge by content matching."""
        all_knowledge = self.load_all_knowledge()

        # Filter by type if specified
        if knowledge_type:
            all_knowledge = [
                k for k in all_knowledge if k.knowledge_type == knowledge_type]

        # Simple text search
        query_lower = query.lower()
        matching_knowledge = []

        for knowledge in all_knowledge:
            if query_lower in knowledge.content.lower(
            ) or query_lower in knowledge.title.lower():
                matching_knowledge.append(knowledge)
                if len(matching_knowledge) >= limit:
                    break

        return matching_knowledge

    def get_knowledge_statistics(self) -> Dict[str, Any]:
        """Get statistics about available psychology knowledge."""
        sources = self.get_available_knowledge_sources()

        stats = {
            "total_sources": len(sources),
            "sources_by_type": {},
            "total_size_bytes": 0,
            "available_types": []
        }

        for source in sources:
            source_type = source["type"]
            if source_type not in stats["sources_by_type"]:
                stats["sources_by_type"][source_type] = 0
                stats["available_types"].append(source_type)

            stats["sources_by_type"][source_type] += 1
            stats["total_size_bytes"] += source.get("size", 0)

        return stats


def main():
    """Test the PsychologyKnowledgeLoader."""
    print("Testing PsychologyKnowledgeLoader...")

    loader = PsychologyKnowledgeLoader()

    print("\nAvailable knowledge sources:")
    sources = loader.get_available_knowledge_sources()
    for source in sources:
        print(f"  - {source['name']} ({source['type']})")

    print("\nKnowledge statistics:")
    stats = loader.get_knowledge_statistics()
    print(f"  Total sources: {stats['total_sources']}")
    print(f"  Available types: {', '.join(stats['available_types'])}")
    print(f"  Total size: {stats['total_size_bytes']:,} bytes")

    print("\nLoading sample knowledge...")
    knowledge_items = loader.load_clinical_knowledge(limit=3)

    if knowledge_items:
        print(f"Loaded {len(knowledge_items)} knowledge items")
        print("\nSample knowledge item:")
        sample = knowledge_items[0]
        print(f"  ID: {sample.id}")
        print(f"  Title: {sample.title}")
        print(f"  Type: {sample.knowledge_type}")
        print(f"  Content length: {sample.content_length} characters")
        print(f"  Has clinical terms: {sample.has_clinical_terms}")
        print(f"  Content preview: {sample.content[:200]}...")
    else:
        print("No knowledge items loaded")


if __name__ == "__main__":
    main()
