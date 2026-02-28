"""
YouTube Transcript Processing and RAG Integration System

This module processes YouTube transcripts from expert creators and creates a
Retrieval-Augmented Generation (RAG) system for dynamic transcript retrieval.

Extended to support knowledge sources (therapeutic books, PDFs, clinical references)
for enhanced RAG retrieval.
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.core.pipelines.processing.nvidia_clients import NemoRetrieverClient

# Centralized output root for runtime artifacts
from ai.core.pipelines.storage_config import get_dataset_pipeline_output_root

# Knowledge sources integration
try:
    from ai.core.pipelines.knowledge_text_extractor import (
        KnowledgeSourceMetadata,
        KnowledgeTextExtractor,
    )

    HAS_KNOWLEDGE_EXTRACTOR = True
except ImportError:
    HAS_KNOWLEDGE_EXTRACTOR = False
    KnowledgeSourceMetadata = None
    KnowledgeTextExtractor = None

# Handle optional dependencies gracefully
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    HAS_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = object  # Dummy class for type hints
    np = None
    cosine_similarity = None
    HAS_TRANSFORMERS = False
    logging.warning(
        "sentence-transformers not installed. RAG search functionality will be limited."
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TranscriptMetadata:
    """Metadata for a YouTube transcript"""

    video_id: str
    title: str
    speaker: str
    duration: float
    language: str
    processed_date: str
    content_hash: str
    word_count: int
    topics: List[str] = field(default_factory=list)
    therapeutic_approaches: List[str] = field(default_factory=list)
    personality_markers: Dict[str, Any] = field(default_factory=dict)
    key_quotes: List[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class RAGIndexEntry:
    """Entry in the RAG index"""

    transcript_id: str
    content: str
    embedding: Optional[List[float]] = None

    metadata: TranscriptMetadata = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class YouTubeRAGSystem:
    """YouTube Transcript Processing and RAG Integration System

    Extended to support knowledge sources (therapeutic books, PDFs,
    clinical references) for enhanced knowledge retrieval.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        include_knowledge_sources: bool = True,
    ):
        # Use environment variable for transcript directory
        # with project-relative default
        transcripts_root = os.getenv("YOUTUBE_TRANSCRIPTS_DIR")
        if transcripts_root:
            self.transcripts_dir = Path(transcripts_root)
        else:
            # Default to project-relative path
            self.transcripts_dir = Path("ai/training_data_consolidated/transcripts")

        self.index_dir = get_dataset_pipeline_output_root() / "rag_index"
        # Create the full directory path if it doesn't exist
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

        # Knowledge sources configuration
        self.include_knowledge_sources = (
            include_knowledge_sources
            and os.getenv("KNOWLEDGE_SOURCES_ENABLED", "true").lower() == "true"
        )
        self.knowledge_extractor = None
        self.knowledge_chunks: Dict[str, List] = {}

        if self.include_knowledge_sources and HAS_KNOWLEDGE_EXTRACTOR:
            try:
                self.knowledge_extractor = KnowledgeTextExtractor()
                logger.info(
                    f"Knowledge sources enabled: "
                    f"{len(self.knowledge_extractor.sources)} sources available"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize knowledge extractor: {e}")
                self.include_knowledge_sources = False

        self.use_nvidia = os.getenv("USE_NVIDIA_RETRIEVER", "false").lower() == "true"
        self.retriever_client = None

        if self.use_nvidia:
            try:
                self.retriever_client = NemoRetrieverClient()
                logger.info("Using NVIDIA NeMo Retriever for embeddings and reranking")
            except Exception as e:
                logger.error(f"Failed to initialize NeMo Retriever: {e}")
                self.use_nvidia = False

        if not self.use_nvidia:
            # Load sentence transformer for embeddings
            try:
                self.encoder = SentenceTransformer(model_name)
                logger.info(f"Loaded sentence transformer: {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {str(e)}")
                self.encoder = None

        self.transcripts: Dict[str, TranscriptMetadata] = {}
        self.rag_index: List[RAGIndexEntry] = []
        self.therapeutic_topics = [
            "complex trauma",
            "ptsd",
            "anxiety",
            "depression",
            "narcissism",
            "codependency",
            "attachment",
            "emotional regulation",
            "dissociation",
            "survival mechanisms",
            "therapeutic techniques",
            "recovery",
        ]
        self.therapeutic_approaches = [
            "cbt",
            "dbt",
            "emdr",
            "trauma-informed",
            "compassion-focused",
            "mindfulness",
            "cognitive restructuring",
            "exposure therapy",
        ]

    def process_transcripts(self) -> Dict[str, TranscriptMetadata]:
        """Process all YouTube transcripts and extract metadata"""
        logger.info("Processing YouTube transcripts...")

        if not self.transcripts_dir.exists():
            logger.warning(f"Transcripts directory not found: {self.transcripts_dir}")
            return {}

        transcript_files = list(self.transcripts_dir.glob("*.md"))
        logger.info(f"Found {len(transcript_files)} transcript files")

        for transcript_file in transcript_files:
            try:
                if metadata := self._extract_transcript_metadata(transcript_file):
                    self.transcripts[metadata.video_id] = metadata
                    logger.info(f"Processed transcript: {metadata.title}")
            except Exception as e:
                logger.error(f"Error processing {transcript_file.name}: {str(e)}")
                continue

        logger.info(f"Processed {len(self.transcripts)} transcripts")
        return self.transcripts

    def _extract_transcript_metadata(
        self, transcript_file: Path
    ) -> Optional[TranscriptMetadata]:
        """Extract metadata from a transcript file"""
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract basic metadata from header
            video_id = transcript_file.stem
            title = self._extract_title(content)
            speaker = self._extract_speaker(content)
            duration = self._extract_duration(content)
            language = self._extract_language(content)
            processed_date = self._extract_processed_date(content)

            # Extract transcript content
            transcript_content = self._extract_transcript_content(content)
            word_count = len(transcript_content.split())

            # Generate content hash
            content_hash = hashlib.md5(transcript_content.encode()).hexdigest()

            # Extract topics and approaches
            topics = self._extract_topics(transcript_content)
            approaches = self._extract_therapeutic_approaches(transcript_content)

            # Extract personality markers
            personality_markers = self._extract_personality_markers(transcript_content)

            # Extract key quotes
            key_quotes = self._extract_key_quotes(transcript_content)

            # Generate summary
            summary = self._generate_summary(transcript_content)

            return TranscriptMetadata(
                video_id=video_id,
                title=title,
                speaker=speaker,
                duration=duration,
                language=language,
                processed_date=processed_date,
                content_hash=content_hash,
                word_count=word_count,
                topics=topics,
                therapeutic_approaches=approaches,
                personality_markers=personality_markers,
                key_quotes=key_quotes[:5],  # Limit to top 5 quotes
                summary=summary,
            )

        except Exception as e:
            logger.error(
                f"Error extracting metadata from {transcript_file.name}: {str(e)}"
            )
            return None

    def _extract_title(self, content: str) -> str:
        """Extract title from transcript content"""
        lines = content.split("\n")
        return next(
            (
                line[2:].split("|")[0].strip()
                for line in lines
                if line.startswith("# ") and "|" in line
            ),
            "Unknown Title",
        )

    def _extract_speaker(self, content: str) -> str:
        """Extract speaker from transcript content"""
        # Look for Tim Fletcher pattern
        if "tim fletcher" in content.lower() or "tim" in content.lower():
            return "Tim Fletcher"
        return "Unknown Speaker"

    def _extract_duration(self, content: str) -> float:
        """Extract duration from transcript content"""
        duration_match = re.search(r"\*\*Duration:\*\* ([\d.]+)", content)
        return float(duration_match[1]) if duration_match else 0.0

    def _extract_language(self, content: str) -> str:
        """Extract language from transcript content"""
        lang_match = re.search(r"\*\*Language:\*\* ([a-z]+)", content)
        return lang_match[1] if lang_match else "en"

    def _extract_processed_date(self, content: str) -> str:
        """Extract processed date from transcript content"""
        date_match = re.search(r"\*\*Processed:\*\* (.+)", content)
        return date_match[1] if date_match else datetime.now(timezone.utc).isoformat()

    def _extract_transcript_content(self, content: str) -> str:
        """Extract actual transcript content"""
        # Find the transcript section
        transcript_start = content.find("## Transcript")
        if transcript_start == -1:
            return content

        return content[transcript_start + len("## Transcript") :].strip()

    def _extract_topics(self, content: str) -> List[str]:
        """Extract therapeutic topics from content"""
        content_lower = content.lower()
        topics_found = [
            topic for topic in self.therapeutic_topics if topic.lower() in content_lower
        ]
        return list(set(topics_found))

    def _extract_therapeutic_approaches(self, content: str) -> List[str]:
        """Extract therapeutic approaches from content"""
        content_lower = content.lower()
        approaches_found = [
            approach
            for approach in self.therapeutic_approaches
            if approach.lower() in content_lower
        ]
        return list(set(approaches_found))

    def _extract_personality_markers(self, content: str) -> Dict[str, Any]:
        """Extract personality markers and speaking style characteristics"""
        return {
            "tone": self._analyze_tone(content),
            "speaking_style": self._analyze_speaking_style(content),
            "emotional_patterns": self._analyze_emotional_patterns(content),
            "communication_approach": self._analyze_communication_approach(content),
        }

    def _analyze_tone(self, content: str) -> str:
        """Analyze the tone of the speaker"""
        compassionate_words = [
            "love",
            "respect",
            "care",
            "understand",
            "empathy",
            "compassion",
        ]
        authoritative_words = ["must", "should", "need", "require", "important"]
        educational_words = ["understand", "learn", "teach", "explain", "knowledge"]

        content_lower = content.lower()
        compassionate_count = sum(word in content_lower for word in compassionate_words)
        authoritative_count = sum(word in content_lower for word in authoritative_words)
        educational_count = sum(word in content_lower for word in educational_words)

        if (
            compassionate_count > authoritative_count
            and compassionate_count > educational_count
        ):
            return "compassionate"
        elif authoritative_count > educational_count:
            return "authoritative"
        else:
            return "educational"

    def _analyze_speaking_style(self, content: str) -> str:
        """Analyze the speaking style"""
        # Look for storytelling patterns
        story_indicators = [
            "so i",
            "let me tell you",
            "for example",
            "imagine",
            "picture this",
        ]
        content_lower = content.lower()

        story_count = sum(indicator in content_lower for indicator in story_indicators)

        if story_count > 3:
            return "storytelling"
        elif len(re.findall(r"\n\s*\n", content)) > 10:  # Many paragraphs
            return "structured"
        else:
            return "conversational"

    def _analyze_emotional_patterns(self, content: str) -> List[str]:
        """Analyze emotional patterns in the content"""
        emotions = []

        if "pain" in content.lower() or "hurt" in content.lower():
            emotions.append("acknowledges_pain")
        if "hope" in content.lower() or "heal" in content.lower():
            emotions.append("offers_hope")
        if "understand" in content.lower() or "realize" in content.lower():
            emotions.append("encourages_insight")
        if "safe" in content.lower() or "protect" in content.lower():
            emotions.append("focuses_on_safety")

        return emotions or ["general_therapeutic"]

    def _analyze_communication_approach(self, content: str) -> str:
        """Analyze the communication approach"""
        if "you can see" in content.lower() or "let me show you" in content.lower():
            return "visual"
        elif "listen" in content.lower() or "hear" in content.lower():
            return "auditory"
        elif "feel" in content.lower() or "experience" in content.lower():
            return "kinesthetic"
        else:
            return "verbal"

    def _extract_key_quotes(self, content: str) -> List[str]:
        """Extract key memorable quotes from the content"""
        # Look for sentences that seem like key insights
        sentences = re.split(r"[.!?]+", content)
        key_quotes = []

        for sentence in sentences:
            sentence = sentence.strip()
            if 50 < len(sentence) < 300 and any(
                keyword in sentence.lower()
                for keyword in [
                    "important to understand",
                    "key",
                    "realize",
                    "understand",
                    "the reality is",
                    "bottom line",
                    "what i want you to understand",
                ]
            ):
                key_quotes.append(sentence)

        return key_quotes[:10]  # Limit to top 10

    def _generate_summary(self, content: str) -> str:
        """Generate a brief summary of the content"""
        # Extract first few meaningful sentences
        sentences = re.split(r"[.!?]+", content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        return (
            " ".join(meaningful_sentences)[:200] + "..."
            if len(" ".join(meaningful_sentences)) > 200
            else " ".join(meaningful_sentences)
        )

    def build_rag_index(self) -> List[RAGIndexEntry]:
        """Build RAG index from processed transcripts and knowledge sources"""
        logger.info("Building RAG index...")
        self._ensure_transcripts_processed()
        self.rag_index = []

        # 1. Process YouTube Transcripts
        self._index_youtube_transcripts()

        # 2. Process Knowledge Sources
        if self.include_knowledge_sources:
            self._index_knowledge_sources()

        logger.info(f"Built RAG index with {len(self.rag_index)} entries")
        return self.rag_index

    def _ensure_transcripts_processed(self):
        """Ensure transcripts are processed before building index"""
        if not self.transcripts:
            logger.warning("No transcripts processed. Processing now...")
            self.process_transcripts()

    def _index_youtube_transcripts(self):
        """Index all processed YouTube transcripts"""
        for video_id, metadata in self.transcripts.items():
            transcript_file = self.transcripts_dir / f"{video_id}.md"
            if transcript_file.exists():
                self._process_single_transcript(video_id, metadata, transcript_file)

    def _process_single_transcript(
        self, video_id: str, metadata: TranscriptMetadata, transcript_file: Path
    ):
        """Process and index a single transcript file"""
        try:
            with open(transcript_file, "r", encoding="utf-8") as f:
                content = f.read()

            transcript_content = self._extract_transcript_content(content)
            chunks = self._chunk_content(transcript_content, max_chunk_size=500)

            for i, chunk in enumerate(chunks):
                entry_id = f"{video_id}_{i}"
                embedding = self._get_embedding(chunk)

                self.rag_index.append(
                    RAGIndexEntry(
                        transcript_id=entry_id,
                        content=chunk,
                        embedding=embedding,
                        metadata=metadata,
                    )
                )
        except Exception as e:
            logger.error(f"Error building index for {video_id}: {str(e)}")

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a single text chunk"""
        if self.use_nvidia and self.retriever_client:
            try:
                return self.retriever_client.get_embedding(text)
            except Exception as e:
                logger.warning(f"NVIDIA Embedding failed: {e}")
        elif self.encoder:
            try:
                return self.encoder.encode(text).tolist()
            except Exception as e:
                logger.warning(f"Failed to generate embedding: {str(e)}")
        return None

    def _index_knowledge_sources(self):
        """Index all available knowledge sources"""
        try:
            knowledge_chunks_list = self._load_knowledge_sources()
            if not knowledge_chunks_list:
                return

            logger.info(f"Retrieved {len(knowledge_chunks_list)} knowledge chunks")

            processed_chunks = []
            chunk_metadatas = []

            for chunk in knowledge_chunks_list:
                processed_chunks.append(chunk.content)
                chunk_metadatas.append(self._create_knowledge_metadata(chunk))

            # Batch process embeddings for knowledge sources
            embeddings = self._get_embeddings_batch(processed_chunks)

            # Create entries
            for chunk_content, meta, emb in zip(
                processed_chunks, chunk_metadatas, embeddings
            ):
                self.rag_index.append(
                    RAGIndexEntry(
                        transcript_id=meta["id"],
                        content=chunk_content,
                        embedding=emb,
                        metadata=meta["metadata"],
                    )
                )
        except Exception as e:
            logger.error(f"Error loading knowledge sources: {e}")

    def _create_knowledge_metadata(self, chunk) -> Dict[str, Any]:
        """Create metadata object for a knowledge base chunk"""
        metadata = TranscriptMetadata(
            video_id=chunk.source_id,
            title=chunk.metadata.title,
            speaker=chunk.metadata.author,
            duration=0.0,
            language="en",
            processed_date=datetime.now(timezone.utc).isoformat(),
            content_hash=chunk.metadata.content_hash or "",
            word_count=len(chunk.content.split()),
            topics=chunk.metadata.topics,
            therapeutic_approaches=[],
            personality_markers={
                "source_type": chunk.metadata.source_type,
                "priority": chunk.metadata.priority,
                "is_knowledge_source": True,
            },
            key_quotes=[],
            summary=f"{chunk.metadata.title} by {chunk.metadata.author}",
        )
        return {"id": chunk.chunk_id, "metadata": metadata}

    def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Get embeddings for a list of texts in batch"""
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        if self.use_nvidia and self.retriever_client:
            return [self._get_embedding(t) for t in texts]
        elif self.encoder:
            try:
                return self.encoder.encode(texts).tolist()
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                return [None] * len(texts)
        return [None] * len(texts)

    def _load_knowledge_sources(self) -> List:
        """Load and return knowledge source chunks."""
        if not self.knowledge_extractor:
            logger.warning("Knowledge extractor not available")
            return []

        logger.info("Loading knowledge sources for RAG integration...")

        # Extract and chunk knowledge sources (prioritize critical and high)
        try:
            chunks_by_source = self.knowledge_extractor.extract_and_chunk_all(
                priority_filter=["critical", "high"]
            )

            all_chunks = []
            for source_id, chunks in chunks_by_source.items():
                all_chunks.extend(chunks)

            return all_chunks
        except Exception as e:
            logger.error(f"Failed to extract knowledge sources: {e}")
            return []

    def _chunk_content(self, content: str, max_chunk_size: int = 500) -> List[str]:
        """Split content into chunks for better retrieval"""
        # Split by paragraphs first
        paragraphs = content.split("\n\n")
        chunks = []

        current_chunk = ""
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) <= max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                if len(paragraph) <= max_chunk_size:
                    current_chunk = paragraph + "\n\n"
                else:
                    # Split long paragraph
                    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_chunk_size:
                            temp_chunk = f"{temp_chunk}{sentence} "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = f"{sentence} "
                    if temp_chunk:
                        chunks.append(temp_chunk.strip())
                    current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def search_transcripts(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search transcripts using semantic similarity"""
        if not self.rag_index:
            logger.warning("RAG index not built. Building now...")
            self.build_rag_index()

        if not self.rag_index:
            return []

        # 1. Get query embedding and optional NVIDIA context
        query_embedding, dual_context = self._prepare_search_context(query, top_k)
        if query_embedding is None or np is None or cosine_similarity is None:
            return self._keyword_search(query, top_k)

        # 2. Calculate similarities
        similarities = self._calculate_similarities(query_embedding)

        # 3. Sort and get top candidates
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_candidates = similarities[: top_k * 2]

        # 4. Apply NVIDIA reranking if enabled
        if (
            self.use_nvidia
            and self.retriever_client
            and (reranked := self._apply_nvidia_reranking(query, top_candidates, top_k))
        ):
            return self._format_reranked_results(reranked, top_candidates, dual_context)

        # 5. Format and return standard results
        return self._format_standard_results(similarities[:top_k], dual_context)

    def _prepare_search_context(self, query: str, top_k: int):
        """Prepare embedding and search context for a query"""
        dual_context = {}
        query_embedding = None

        if self.use_nvidia and self.retriever_client:
            try:
                logger.info(f"Using NVIDIA Tri-Persona Search for: {query}")
                candidates = [entry.content for entry in self.rag_index[:100]]
                dual_context = self.retriever_client.tri_persona_search(
                    query, documents=candidates, top_k=top_k
                )
                query_embedding = self.retriever_client.get_embedding(query)
            except Exception as e:
                logger.error(f"NVIDIA Dual Persona Search failed: {e}")
        elif self.encoder:
            try:
                query_embedding = self.encoder.encode(query)
            except Exception as e:
                logger.error(f"Failed to encode query: {str(e)}")

        return query_embedding, dual_context

    def _calculate_similarities(self, query_embedding: Any) -> List[tuple]:
        """Calculate cosine similarities between query and index"""
        similarities = []
        q_emb_arr = np.array(query_embedding).reshape(1, -1)

        for entry in self.rag_index:
            if entry.embedding:
                try:
                    similarity = cosine_similarity(
                        q_emb_arr, np.array(entry.embedding).reshape(1, -1)
                    )[0][0]
                    similarities.append((entry, similarity))
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {str(e)}")
        return similarities

    def _apply_nvidia_reranking(self, query: str, top_candidates: List, top_k: int):
        """Apply NVIDIA safety-constrained reranking"""
        try:
            candidate_texts = [entry.content for entry, _ in top_candidates]
            return self.retriever_client.safety_constrained_rerank(
                query, documents=candidate_texts, top_n=top_k
            )
        except Exception as e:
            logger.error(
                f"NVIDIA Safety Rerank failed: {e}. Returning raw similarity results."
            )
            return None

    def _format_reranked_results(
        self, reranked: List, top_candidates: List, dual_context: Dict
    ):
        """Format reranked search results"""
        results = []
        for item in reranked:
            content = item.get("text", item.get("document", ""))
            for entry, sim in top_candidates:
                if entry.content == content:
                    results.append(
                        {
                            "content": entry.content,
                            "similarity": float(item.get("relevance_score", sim)),
                            "metadata": {
                                **self._get_basic_metadata(entry),
                                "dual_persona_context": {} if results else dual_context,
                                "safety_reranked": True,
                            },
                            "transcript_id": entry.transcript_id,
                        }
                    )
                    break
        return results

    def _format_standard_results(self, top_similarities: List, dual_context: Dict):
        """Format standard search results"""
        return [
            {
                "content": entry.content,
                "similarity": float(similarity),
                "metadata": {
                    **self._get_basic_metadata(entry),
                    "personality_markers": entry.metadata.personality_markers,
                    "dual_persona_context": dual_context if i == 0 else {},
                },
                "transcript_id": entry.transcript_id,
            }
            for i, (entry, similarity) in enumerate(top_similarities)
        ]

    def _get_basic_metadata(self, entry: RAGIndexEntry) -> Dict[str, Any]:
        """Get basic metadata for a result entry"""
        return {
            "title": entry.metadata.title,
            "speaker": entry.metadata.speaker,
            "topics": entry.metadata.topics,
            "therapeutic_approaches": entry.metadata.therapeutic_approaches,
            "summary": entry.metadata.summary,
        }

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        results = []

        for entry in self.rag_index:
            content_lower = entry.content.lower()
            # Simple keyword matching score
            score = sum(word in content_lower for word in query_lower.split())

            if score > 0:
                results.append((entry, score))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)

        formatted_results = []
        for entry, score in results[:top_k]:
            result = {
                "content": entry.content,
                "similarity": float(score) / len(query.split()),  # Normalize
                "metadata": {
                    "title": entry.metadata.title,
                    "speaker": entry.metadata.speaker,
                    "topics": entry.metadata.topics,
                    "therapeutic_approaches": entry.metadata.therapeutic_approaches,
                    "summary": entry.metadata.summary,
                },
                "transcript_id": entry.transcript_id,
            }
            formatted_results.append(result)

        return formatted_results

    def get_few_shot_examples(self, topic: str, count: int = 3) -> List[Dict[str, Any]]:
        """Get few-shot examples for a specific therapeutic topic"""
        examples = []

        if not self.rag_index:
            self.build_rag_index()

        # Find entries related to the topic
        topic_lower = topic.lower()
        relevant_entries = []

        relevant_entries = [
            entry
            for entry in self.rag_index
            if topic_lower in entry.content.lower()
            or any(topic_lower in t.lower() for t in entry.metadata.topics)
        ]

        # Select diverse examples
        selected_entries = (
            relevant_entries[:count]
            if len(relevant_entries) >= count
            else relevant_entries
        )

        for entry in selected_entries:
            example = {
                "input": f"Client is struggling with {topic}",
                "output": f"{entry.content[:300]}..."
                if len(entry.content) > 300
                else entry.content,
                "context": {
                    "speaker": entry.metadata.speaker,
                    "title": entry.metadata.title,
                    "therapeutic_approaches": entry.metadata.therapeutic_approaches,
                    "key_insights": entry.metadata.key_quotes[:2],
                },
            }
            examples.append(example)

        return examples

    def save_index(self):
        """Save the RAG index to disk"""
        index_file = self.index_dir / "youtube_rag_index.json"

        # Convert index to serializable format
        serializable_index = []
        for entry in self.rag_index:
            serializable_entry = {
                "transcript_id": entry.transcript_id,
                "content": entry.content,
                "embedding": entry.embedding,
                "metadata": {
                    "video_id": entry.metadata.video_id,
                    "title": entry.metadata.title,
                    "speaker": entry.metadata.speaker,
                    "duration": entry.metadata.duration,
                    "language": entry.metadata.language,
                    "processed_date": entry.metadata.processed_date,
                    "content_hash": entry.metadata.content_hash,
                    "word_count": entry.metadata.word_count,
                    "topics": entry.metadata.topics,
                    "therapeutic_approaches": entry.metadata.therapeutic_approaches,
                    "personality_markers": entry.metadata.personality_markers,
                    "key_quotes": entry.metadata.key_quotes,
                    "summary": entry.metadata.summary,
                },
                "timestamp": entry.timestamp,
            }
            serializable_index.append(serializable_entry)

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(serializable_index, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved RAG index to {index_file}")

    def load_index(self):
        """Load the RAG index from disk"""
        index_file = self.index_dir / "youtube_rag_index.json"

        if not index_file.exists():
            logger.warning(f"RAG index file not found: {index_file}")
            return

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                serializable_index = json.load(f)

            self.rag_index = []
            for entry_data in serializable_index:
                metadata = TranscriptMetadata(
                    video_id=entry_data["metadata"]["video_id"],
                    title=entry_data["metadata"]["title"],
                    speaker=entry_data["metadata"]["speaker"],
                    duration=entry_data["metadata"]["duration"],
                    language=entry_data["metadata"]["language"],
                    processed_date=entry_data["metadata"]["processed_date"],
                    content_hash=entry_data["metadata"]["content_hash"],
                    word_count=entry_data["metadata"]["word_count"],
                    topics=entry_data["metadata"]["topics"],
                    therapeutic_approaches=entry_data["metadata"][
                        "therapeutic_approaches"
                    ],
                    personality_markers=entry_data["metadata"]["personality_markers"],
                    key_quotes=entry_data["metadata"]["key_quotes"],
                    summary=entry_data["metadata"]["summary"],
                )

                entry = RAGIndexEntry(
                    transcript_id=entry_data["transcript_id"],
                    content=entry_data["content"],
                    embedding=entry_data["embedding"],
                    metadata=metadata,
                    timestamp=entry_data["timestamp"],
                )

                self.rag_index.append(entry)

            logger.info(f"Loaded RAG index with {len(self.rag_index)} entries")

        except Exception as e:
            logger.error(f"Failed to load RAG index: {str(e)}")

    def get_transcript_statistics(self) -> Dict[str, Any]:
        """Get statistics about processed transcripts"""
        if not self.transcripts:
            self.process_transcripts()

        total_duration = sum(t.duration for t in self.transcripts.values())
        total_words = sum(t.word_count for t in self.transcripts.values())
        topics_count = {}
        approaches_count = {}
        speakers = set()

        for transcript in self.transcripts.values():
            speakers.add(transcript.speaker)
            for topic in transcript.topics:
                topics_count[topic] = topics_count.get(topic, 0) + 1
            for approach in transcript.therapeutic_approaches:
                approaches_count[approach] = approaches_count.get(approach, 0) + 1

        return {
            "total_transcripts": len(self.transcripts),
            "total_speakers": len(speakers),
            "speakers": list(speakers),
            "total_duration_hours": round(total_duration / 3600, 2),
            "total_words": total_words,
            "average_words_per_transcript": round(total_words / len(self.transcripts))
            if self.transcripts
            else 0,
            "topics_distribution": topics_count,
            "approaches_distribution": approaches_count,
            "indexed_chunks": len(self.rag_index),
        }


# Convenience functions
def create_youtube_rag_system() -> YouTubeRAGSystem:
    """Create and initialize YouTube RAG system"""
    return YouTubeRAGSystem()


def process_all_transcripts() -> YouTubeRAGSystem:
    """Process all transcripts and build RAG index"""
    system = create_youtube_rag_system()
    system.process_transcripts()
    system.build_rag_index()
    system.save_index()
    return system


def search_therapeutic_content(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search therapeutic content from YouTube transcripts"""
    system = create_youtube_rag_system()
    system.load_index()
    return system.search_transcripts(query, top_k)


def get_few_shot_examples(topic: str, count: int = 3) -> List[Dict[str, Any]]:
    """Get few-shot examples for therapeutic scenarios"""
    system = create_youtube_rag_system()
    system.load_index()
    return system.get_few_shot_examples(topic, count)


if __name__ == "__main__":
    # Example usage
    try:
        print("Processing YouTube transcripts and building RAG system...")
        system = process_all_transcripts()

        # Show statistics
        stats = system.get_transcript_statistics()
        print("\nTranscript Statistics:")
        print(f"  Total transcripts: {stats['total_transcripts']}")
        print(f"  Total duration: {stats['total_duration_hours']} hours")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Indexed chunks: {stats['indexed_chunks']}")
        print(f"  Speakers: {', '.join(stats['speakers'])}")

        # Example search
        print("\nExample search for 'complex trauma':")
        results = system.search_transcripts("complex trauma", top_k=2)
        for i, result in enumerate(results, 1):
            print(
                f"  {i}. {result['metadata']['title']} "
                f"(similarity: {result['similarity']:.3f})"
            )
            print(f"     Content preview: {result['content'][:100]}...")
            print()

    except Exception as e:
        print(f"Error: {str(e)}")
        raise
