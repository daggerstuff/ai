"""
YouTube Transcript Processing and RAG Integration System

This module processes YouTube transcripts from expert creators and creates a
Retrieval-Augmented Generation (RAG) system for dynamic transcript retrieval.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Centralized output root for runtime artifacts
from ai.dataset_pipeline.storage_config import get_dataset_pipeline_output_root

# Handle optional dependencies gracefully
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_TRANSFORMERS = True
except ImportError:
    SentenceTransformer = object  # Dummy class for type hints
    np = None
    cosine_similarity = None
    HAS_TRANSFORMERS = False
    logging.warning("sentence-transformers not installed. RAG search functionality will be limited.")

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
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class YouTubeRAGSystem:
    """YouTube Transcript Processing and RAG Integration System"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.transcripts_dir = Path("ai/training_data_consolidated/transcripts")
        self.index_dir = get_dataset_pipeline_output_root() / "rag_index"
        # Create the full directory path if it doesn't exist
        self.index_dir.parent.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(exist_ok=True)

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
            "complex trauma", "ptsd", "anxiety", "depression", "narcissism",
            "codependency", "attachment", "emotional regulation", "dissociation",
            "survival mechanisms", "therapeutic techniques", "recovery"
        ]
        self.therapeutic_approaches = [
            "cbt", "dbt", "emdr", "trauma-informed", "compassion-focused",
            "mindfulness", "cognitive restructuring", "exposure therapy"
        ]

    def process_transcripts(self) -> Dict[str, TranscriptMetadata]:
        """Process all YouTube transcripts and extract metadata"""
        logger.info("Processing YouTube transcripts...")

        if not self.transcripts_dir.exists():
            logger.error(f"Transcripts directory not found: {self.transcripts_dir}")
            return {}

        transcript_files = list(self.transcripts_dir.glob("*.md"))
        logger.info(f"Found {len(transcript_files)} transcript files")

        for transcript_file in transcript_files:
            try:
                metadata = self._extract_transcript_metadata(transcript_file)
                if metadata:
                    self.transcripts[metadata.video_id] = metadata
                    logger.info(f"Processed transcript: {metadata.title}")
            except Exception as e:
                logger.error(f"Error processing {transcript_file.name}: {str(e)}")
                continue

        logger.info(f"Processed {len(self.transcripts)} transcripts")
        return self.transcripts

    def _extract_transcript_metadata(self, transcript_file: Path) -> Optional[TranscriptMetadata]:
        """Extract metadata from a transcript file"""
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
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

            metadata = TranscriptMetadata(
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
                summary=summary
            )

            return metadata

        except Exception as e:
            logger.error(f"Error extracting metadata from {transcript_file.name}: {str(e)}")
            return None

    def _extract_title(self, content: str) -> str:
        """Extract title from transcript content"""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# ') and '|' in line:
                return line[2:].split('|')[0].strip()
        return "Unknown Title"

    def _extract_speaker(self, content: str) -> str:
        """Extract speaker from transcript content"""
        # Look for Tim Fletcher pattern
        if "tim fletcher" in content.lower() or "tim" in content.lower():
            return "Tim Fletcher"
        return "Unknown Speaker"

    def _extract_duration(self, content: str) -> float:
        """Extract duration from transcript content"""
        duration_match = re.search(r'**Duration:** ([\d.]+)', content)
        if duration_match:
            return float(duration_match.group(1))
        return 0.0

    def _extract_language(self, content: str) -> str:
        """Extract language from transcript content"""
        lang_match = re.search(r'**Language:** ([a-z]+)', content)
        if lang_match:
            return lang_match.group(1)
        return "en"

    def _extract_processed_date(self, content: str) -> str:
        """Extract processed date from transcript content"""
        date_match = re.search(r'**Processed:** (.+)', content)
        if date_match:
            return date_match.group(1)
        return datetime.utcnow().isoformat()

    def _extract_transcript_content(self, content: str) -> str:
        """Extract actual transcript content"""
        # Find the transcript section
        transcript_start = content.find('## Transcript')
        if transcript_start == -1:
            return content

        transcript_content = content[transcript_start + len('## Transcript'):].strip()
        return transcript_content

    def _extract_topics(self, content: str) -> List[str]:
        """Extract therapeutic topics from content"""
        topics_found = []
        content_lower = content.lower()

        for topic in self.therapeutic_topics:
            if topic.lower() in content_lower:
                topics_found.append(topic)

        return list(set(topics_found))

    def _extract_therapeutic_approaches(self, content: str) -> List[str]:
        """Extract therapeutic approaches from content"""
        approaches_found = []
        content_lower = content.lower()

        for approach in self.therapeutic_approaches:
            if approach.lower() in content_lower:
                approaches_found.append(approach)

        return list(set(approaches_found))

    def _extract_personality_markers(self, content: str) -> Dict[str, Any]:
        """Extract personality markers and speaking style characteristics"""
        markers = {
            "tone": self._analyze_tone(content),
            "speaking_style": self._analyze_speaking_style(content),
            "emotional_patterns": self._analyze_emotional_patterns(content),
            "communication_approach": self._analyze_communication_approach(content)
        }
        return markers

    def _analyze_tone(self, content: str) -> str:
        """Analyze the tone of the speaker"""
        compassionate_words = ["love", "respect", "care", "understand", "empathy", "compassion"]
        authoritative_words = ["must", "should", "need", "require", "important"]
        educational_words = ["understand", "learn", "teach", "explain", "knowledge"]

        content_lower = content.lower()
        compassionate_count = sum(1 for word in compassionate_words if word in content_lower)
        authoritative_count = sum(1 for word in authoritative_words if word in content_lower)
        educational_count = sum(1 for word in educational_words if word in content_lower)

        if compassionate_count > authoritative_count and compassionate_count > educational_count:
            return "compassionate"
        elif authoritative_count > educational_count:
            return "authoritative"
        else:
            return "educational"

    def _analyze_speaking_style(self, content: str) -> str:
        """Analyze the speaking style"""
        # Look for storytelling patterns
        story_indicators = ["so i", "let me tell you", "for example", "imagine", "picture this"]
        content_lower = content.lower()

        story_count = sum(1 for indicator in story_indicators if indicator in content_lower)

        if story_count > 3:
            return "storytelling"
        elif len(re.findall(r'\n\s*\n', content)) > 10:  # Many paragraphs
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

        return emotions if emotions else ["general_therapeutic"]

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
        sentences = re.split(r'[.!?]+', content)
        key_quotes = []

        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 50 and len(sentence) < 300:  # Reasonable quote length
                # Look for insightful or impactful statements
                if any(keyword in sentence.lower() for keyword in [
                    "important to understand", "key", "realize", "understand",
                    "the reality is", "bottom line", "what i want you to understand"
                ]):
                    key_quotes.append(sentence)

        return key_quotes[:10]  # Limit to top 10

    def _generate_summary(self, content: str) -> str:
        """Generate a brief summary of the content"""
        # Extract first few meaningful sentences
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:3]
        return " ".join(meaningful_sentences)[:200] + "..." if len(" ".join(meaningful_sentences)) > 200 else " ".join(meaningful_sentences)

    def build_rag_index(self) -> List[RAGIndexEntry]:
        """Build RAG index from processed transcripts"""
        logger.info("Building RAG index...")

        if not self.transcripts:
            logger.warning("No transcripts processed. Processing now...")
            self.process_transcripts()

        self.rag_index = []

        for video_id, metadata in self.transcripts.items():
            transcript_file = self.transcripts_dir / f"{video_id}.md"
            if transcript_file.exists():
                try:
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    transcript_content = self._extract_transcript_content(content)

                    # Split content into chunks for better retrieval
                    chunks = self._chunk_content(transcript_content, max_chunk_size=500)

                    for i, chunk in enumerate(chunks):
                        entry_id = f"{video_id}_{i}"

                        # Generate embedding if encoder is available
                        embedding = None
                        if self.encoder:
                            try:
                                embedding = self.encoder.encode(chunk).tolist()
                            except Exception as e:
                                logger.warning(f"Failed to generate embedding for {entry_id}: {str(e)}")

                        entry = RAGIndexEntry(
                            transcript_id=entry_id,
                            content=chunk,
                            embedding=embedding,
                            metadata=metadata
                        )

                        self.rag_index.append(entry)

                except Exception as e:
                    logger.error(f"Error building index for {video_id}: {str(e)}")
                    continue

        logger.info(f"Built RAG index with {len(self.rag_index)} entries")
        return self.rag_index

    def _chunk_content(self, content: str, max_chunk_size: int = 500) -> List[str]:
        """Split content into chunks for better retrieval"""
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
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
                    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                    temp_chunk = ""
                    for sentence in sentences:
                        if len(temp_chunk) + len(sentence) <= max_chunk_size:
                            temp_chunk += sentence + " "
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence + " "
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

        # Generate query embedding
        if self.encoder:
            try:
                query_embedding = self.encoder.encode(query).reshape(1, -1)
            except Exception as e:
                logger.error(f"Failed to encode query: {str(e)}")
                return []
        else:
            # Fallback to keyword matching
            return self._keyword_search(query, top_k)

        # Calculate similarities
        similarities = []
        for entry in self.rag_index:
            if entry.embedding:
                try:
                    similarity = cosine_similarity(query_embedding, np.array(entry.embedding).reshape(1, -1))[0][0]
                    similarities.append((entry, similarity))
                except Exception as e:
                    logger.warning(f"Error calculating similarity: {str(e)}")
                    continue

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []

        for entry, similarity in similarities[:top_k]:
            result = {
                "content": entry.content,
                "similarity": float(similarity),
                "metadata": {
                    "title": entry.metadata.title,
                    "speaker": entry.metadata.speaker,
                    "topics": entry.metadata.topics,
                    "therapeutic_approaches": entry.metadata.therapeutic_approaches,
                    "summary": entry.metadata.summary
                },
                "transcript_id": entry.transcript_id
            }
            results.append(result)

        return results

    def _keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search"""
        query_lower = query.lower()
        results = []

        for entry in self.rag_index:
            content_lower = entry.content.lower()
            # Simple keyword matching score
            score = sum(1 for word in query_lower.split() if word in content_lower)

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
                    "summary": entry.metadata.summary
                },
                "transcript_id": entry.transcript_id
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

        for entry in self.rag_index:
            if (topic_lower in entry.content.lower() or
                any(topic_lower in t.lower() for t in entry.metadata.topics)):
                relevant_entries.append(entry)

        # Select diverse examples
        selected_entries = relevant_entries[:count] if len(relevant_entries) >= count else relevant_entries

        for entry in selected_entries:
            example = {
                "input": f"Client is struggling with {topic}",
                "output": entry.content[:300] + "..." if len(entry.content) > 300 else entry.content,
                "context": {
                    "speaker": entry.metadata.speaker,
                    "title": entry.metadata.title,
                    "therapeutic_approaches": entry.metadata.therapeutic_approaches,
                    "key_insights": entry.metadata.key_quotes[:2]
                }
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
                    "summary": entry.metadata.summary
                },
                "timestamp": entry.timestamp
            }
            serializable_index.append(serializable_entry)

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_index, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved RAG index to {index_file}")

    def load_index(self):
        """Load the RAG index from disk"""
        index_file = self.index_dir / "youtube_rag_index.json"

        if not index_file.exists():
            logger.warning(f"RAG index file not found: {index_file}")
            return

        try:
            with open(index_file, 'r', encoding='utf-8') as f:
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
                    therapeutic_approaches=entry_data["metadata"]["therapeutic_approaches"],
                    personality_markers=entry_data["metadata"]["personality_markers"],
                    key_quotes=entry_data["metadata"]["key_quotes"],
                    summary=entry_data["metadata"]["summary"]
                )

                entry = RAGIndexEntry(
                    transcript_id=entry_data["transcript_id"],
                    content=entry_data["content"],
                    embedding=entry_data["embedding"],
                    metadata=metadata,
                    timestamp=entry_data["timestamp"]
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
            "average_words_per_transcript": round(total_words / len(self.transcripts)) if self.transcripts else 0,
            "topics_distribution": topics_count,
            "approaches_distribution": approaches_count,
            "indexed_chunks": len(self.rag_index)
        }

# Convenience functions
def create_youtube_rag_system() -> YouTubeRAGSystem:
    """Create and initialize YouTube RAG system"""
    system = YouTubeRAGSystem()
    return system

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
        print(f"\nTranscript Statistics:")
        print(f"  Total transcripts: {stats['total_transcripts']}")
        print(f"  Total duration: {stats['total_duration_hours']} hours")
        print(f"  Total words: {stats['total_words']:,}")
        print(f"  Indexed chunks: {stats['indexed_chunks']}")
        print(f"  Speakers: {', '.join(stats['speakers'])}")

        # Example search
        print(f"\nExample search for 'complex trauma':")
        results = system.search_transcripts("complex trauma", top_k=2)
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['metadata']['title']} (similarity: {result['similarity']:.3f})")
            print(f"     Content preview: {result['content'][:100]}...")
            print()

    except Exception as e:
        print(f"Error: {str(e)}")
        raise