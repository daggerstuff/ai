"""
Knowledge Text Extractor Module
-------------------------------
Extracts text from various book formats (EPUB, PDF, AZW3) for RAG integration.

This module provides utilities to:
1. Download knowledge sources from S3
2. Extract text using calibre (ebook-convert) or pdftotext
3. Chunk content for embedding generation
4. Cache extracted text locally for performance
"""

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KnowledgeSourceMetadata:
    """Metadata for a knowledge source (book, PDF, clinical reference)."""

    source_id: str
    title: str
    author: str
    source_type: str  # "therapeutic_book", "clinical_reference"
    topics: List[str] = field(default_factory=list)
    priority: str = "medium"
    file_path: str = ""
    content_hash: str = ""
    word_count: int = 0
    extraction_method: str = ""
    size_bytes: int = 0


@dataclass
class ExtractedChunk:
    """A chunk of extracted text from a knowledge source."""

    chunk_id: str
    content: str
    source_id: str
    chunk_index: int
    metadata: KnowledgeSourceMetadata


class KnowledgeTextExtractor:
    """Extracts and processes text from knowledge sources for RAG integration."""

    def __init__(
        self,
        registry_path: str = "ai/data/knowledge_sources_registry.json",
        cache_dir: Optional[str] = None,
    ):
        self.registry_path = Path(registry_path)
        self.cache_dir = Path(
            cache_dir
            or os.getenv("KNOWLEDGE_SOURCES_LOCAL_CACHE", "/tmp/knowledge_sources")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.text_cache_dir = self.cache_dir / "extracted_text"
        self.text_cache_dir.mkdir(parents=True, exist_ok=True)

        self.registry: Dict = {}
        self.sources: Dict[str, KnowledgeSourceMetadata] = {}

        # Chunk configuration (matching YouTubeRAGSystem)
        self.chunk_size = 500  # tokens (approximated as words)
        self.chunk_overlap = 50

        self._load_registry()

    def _load_registry(self) -> None:
        """Load the knowledge sources registry."""
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return

        with open(self.registry_path, "r") as f:
            self.registry = json.load(f)

        self._parse_sources()

    def _parse_sources(self) -> None:
        """Parse knowledge sources from registry into metadata objects."""
        knowledge_sources = self.registry.get("knowledge_sources", {})

        # Parse therapeutic books
        for source_id, info in knowledge_sources.get("therapeutic_books", {}).items():
            self.sources[source_id] = KnowledgeSourceMetadata(
                source_id=source_id,
                title=info.get("title", "Unknown Title"),
                author=info.get("author", "Unknown Author"),
                source_type="therapeutic_book",
                topics=info.get("topics", []),
                priority=info.get("priority", "medium"),
                file_path=info.get("path", ""),
                extraction_method=info.get("extraction_method", ""),
                size_bytes=info.get("size_bytes", 0),
            )

        # Parse clinical references
        for source_id, info in knowledge_sources.get("clinical_references", {}).items():
            self.sources[source_id] = KnowledgeSourceMetadata(
                source_id=source_id,
                title=info.get("title", "Unknown Title"),
                author=info.get("author", "Unknown Author"),
                source_type="clinical_reference",
                topics=info.get("topics", []),
                priority=info.get("priority", "medium"),
                file_path=info.get("path", ""),
                extraction_method=info.get("extraction_method", ""),
                size_bytes=info.get("size_bytes", 0),
            )

        # Parse already integrated sources
        for source_id, info in knowledge_sources.get(
            "existing_integrated_pdfs", {}
        ).items():
            self.sources[source_id] = KnowledgeSourceMetadata(
                source_id=source_id,
                title=info.get("title", "Unknown Title"),
                author=info.get("author", "Unknown Author"),
                source_type="integrated_pdf",
                topics=info.get("topics", []),
                priority=info.get("priority", "medium"),
                file_path=info.get("path", ""),
                extraction_method=info.get("extraction_method", ""),
                size_bytes=info.get("size_bytes", 0),
            )

        logger.info(f"Parsed {len(self.sources)} knowledge sources from registry")

    def get_cached_text_path(self, source_id: str) -> Path:
        """Get the path where extracted text is cached."""
        return self.text_cache_dir / f"{source_id}.txt"

    def is_text_cached(self, source_id: str) -> bool:
        """Check if text has already been extracted and cached."""
        return self.get_cached_text_path(source_id).exists()

    def extract_text(self, source_id: str) -> Optional[str]:
        """
        Extract text from a knowledge source.

        Downloads from S3 if needed, extracts text using appropriate tool,
        and caches the result locally.
        """
        if source_id not in self.sources:
            logger.error(f"Unknown source: {source_id}")
            return None

        source = self.sources[source_id]

        # Check cache first
        cached_path = self.get_cached_text_path(source_id)
        if cached_path.exists():
            logger.info(f"Using cached text for {source_id}")
            return cached_path.read_text(encoding="utf-8")

        # Download from S3 and extract
        extraction_method = source.extraction_method

        if extraction_method == "epub_to_text":
            text = self._extract_epub(source)
        elif extraction_method == "pdf_to_text":
            text = self._extract_pdf(source)
        elif extraction_method == "azw3_to_text":
            text = self._extract_azw3(source)
        else:
            logger.error(f"Unknown extraction method: {extraction_method}")
            return None

        if text:
            # Cache the extracted text
            cached_path.write_text(text, encoding="utf-8")
            logger.info(f"Cached extracted text for {source_id}")

            # Update metadata
            source.content_hash = hashlib.md5(text.encode()).hexdigest()
            source.word_count = len(text.split())

        return text

    def _download_from_s3(self, s3_path: str, local_path: Path) -> bool:
        """Download a file from S3 using rclone."""
        # Convert s3://pixel-data/path to pixel-data:path
        rclone_path = s3_path.replace("s3://pixel-data/", "pixel-data:")
        if not rclone_path.startswith("pixel-data:"):
            # Handle other buckets if any
            rclone_path = s3_path.replace("s3://", "").replace("/", ":", 1)

        cmd = ["rclone", "copyto", rclone_path, str(local_path), "--progress"]

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to download from S3: {e.stderr}")
            return False
        except FileNotFoundError:
            logger.error("rclone not found. Please install rclone.")
            return False

    def _extract_epub(self, source: KnowledgeSourceMetadata) -> Optional[str]:
        """Extract text from EPUB using calibre's ebook-convert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            filename = Path(source.file_path).name
            local_epub = tmp_path / filename
            local_txt = tmp_path / f"{source.source_id}.txt"

            # Download from S3
            if not self._download_from_s3(source.file_path, local_epub):
                return None

            # Find the actual downloaded file
            epub_files = list(tmp_path.glob("*.epub"))
            if not epub_files:
                logger.error(f"EPUB not found after download: {source.source_id}")
                return None

            local_epub = epub_files[0]

            # Convert using calibre
            try:
                subprocess.run(
                    [
                        "ebook-convert",
                        str(local_epub),
                        str(local_txt),
                        "--enable-heuristics",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return local_txt.read_text(encoding="utf-8")
            except subprocess.CalledProcessError as e:
                logger.error(f"ebook-convert failed for {source.source_id}: {e.stderr}")
                return None
            except FileNotFoundError:
                logger.error("ebook-convert not found. Please install calibre.")
                return None

    def _extract_pdf(self, source: KnowledgeSourceMetadata) -> Optional[str]:
        """Extract text from PDF using pdftotext."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            filename = Path(source.file_path).name
            local_pdf = tmp_path / filename
            local_txt = tmp_path / f"{source.source_id}.txt"

            # Download from S3
            if not self._download_from_s3(source.file_path, local_pdf):
                return None

            # Find the actual downloaded file
            pdf_files = list(tmp_path.glob("*.pdf"))
            if not pdf_files:
                logger.error(f"PDF not found after download: {source.source_id}")
                return None

            local_pdf = pdf_files[0]

            # Extract using pdftotext
            try:
                subprocess.run(
                    ["pdftotext", "-layout", str(local_pdf), str(local_txt)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return local_txt.read_text(encoding="utf-8")
            except subprocess.CalledProcessError as e:
                logger.error(f"pdftotext failed for {source.source_id}: {e.stderr}")
                return None
            except FileNotFoundError:
                logger.error("pdftotext not found. Please install poppler-utils.")
                return None

    def _extract_azw3(self, source: KnowledgeSourceMetadata) -> Optional[str]:
        """Extract text from AZW3 (Kindle format) using calibre's ebook-convert."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            filename = Path(source.file_path).name
            local_azw3 = tmp_path / filename
            local_txt = tmp_path / f"{source.source_id}.txt"

            # Download from S3
            if not self._download_from_s3(source.file_path, local_azw3):
                return None

            # Find the actual downloaded file
            azw3_files = list(tmp_path.glob("*.azw3"))
            if not azw3_files:
                logger.error(f"AZW3 not found after download: {source.source_id}")
                return None

            local_azw3 = azw3_files[0]

            # Convert using calibre
            try:
                subprocess.run(
                    [
                        "ebook-convert",
                        str(local_azw3),
                        str(local_txt),
                        "--enable-heuristics",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                return local_txt.read_text(encoding="utf-8")
            except subprocess.CalledProcessError as e:
                logger.error(f"ebook-convert failed for {source.source_id}: {e.stderr}")
                return None
            except FileNotFoundError:
                logger.error("ebook-convert not found. Please install calibre.")
                return None

    def chunk_text(self, text: str, source_id: str) -> List[ExtractedChunk]:
        """
        Split text into chunks suitable for embedding.

        Uses paragraph-aware chunking similar to YouTubeRAGSystem.
        """
        if source_id not in self.sources:
            return []

        source = self.sources[source_id]
        chunks: List[ExtractedChunk] = []

        # Split by paragraphs
        paragraphs = text.split("\n\n")
        current_chunk = ""
        chunk_index = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            # Check if adding this paragraph exceeds chunk size
            word_count = len((current_chunk + " " + paragraph).split())

            if word_count <= self.chunk_size:
                current_chunk = (current_chunk + "\n\n" + paragraph).strip()
            else:
                # Save current chunk if non-empty
                if current_chunk:
                    chunk = ExtractedChunk(
                        chunk_id=f"{source_id}_{chunk_index}",
                        content=current_chunk,
                        source_id=source_id,
                        chunk_index=chunk_index,
                        metadata=source,
                    )
                    chunks.append(chunk)
                    chunk_index += 1

                # Start new chunk (with overlap from previous)
                if len(paragraph.split()) <= self.chunk_size:
                    # Add overlap from previous chunk
                    overlap_words = current_chunk.split()[-self.chunk_overlap :]
                    current_chunk = " ".join(overlap_words) + "\n\n" + paragraph
                else:
                    # Paragraph too long, split it
                    sentences = paragraph.replace(". ", ".\n").split("\n")
                    current_chunk = ""
                    for sentence in sentences:
                        if (
                            len((current_chunk + " " + sentence).split())
                            <= self.chunk_size
                        ):
                            current_chunk = (current_chunk + " " + sentence).strip()
                        else:
                            if current_chunk:
                                chunk = ExtractedChunk(
                                    chunk_id=f"{source_id}_{chunk_index}",
                                    content=current_chunk,
                                    source_id=source_id,
                                    chunk_index=chunk_index,
                                    metadata=source,
                                )
                                chunks.append(chunk)
                                chunk_index += 1
                            current_chunk = sentence

        # Don't forget the last chunk
        if current_chunk:
            chunk = ExtractedChunk(
                chunk_id=f"{source_id}_{chunk_index}",
                content=current_chunk,
                source_id=source_id,
                chunk_index=chunk_index,
                metadata=source,
            )
            chunks.append(chunk)

        logger.info(f"Created {len(chunks)} chunks from {source_id}")
        return chunks

    def extract_and_chunk_all(
        self, priority_filter: Optional[List[str]] = None
    ) -> Dict[str, List[ExtractedChunk]]:
        """
        Extract and chunk all knowledge sources.

        Args:
            priority_filter: If provided, only extract sources with matching priority
                           e.g., ["critical", "high"]

        Returns:
            Dict mapping source_id to list of ExtractedChunks
        """
        all_chunks: Dict[str, List[ExtractedChunk]] = {}

        for source_id, source in self.sources.items():
            # Skip if priority doesn't match filter
            if priority_filter and source.priority not in priority_filter:
                logger.info(f"Skipping {source_id} (priority: {source.priority})")
                continue

            # Skip already integrated sources (they're in training path)
            if source.source_type == "integrated_pdf":
                logger.info(f"Skipping {source_id} (already integrated)")
                continue

            logger.info(f"Processing {source_id}: {source.title}")

            text = self.extract_text(source_id)
            if text:
                chunks = self.chunk_text(text, source_id)
                all_chunks[source_id] = chunks
            else:
                logger.warning(f"Failed to extract text from {source_id}")

        total_chunks = sum(len(chunks) for chunks in all_chunks.values())
        logger.info(
            f"Extracted {total_chunks} total chunks from {len(all_chunks)} sources"
        )

        return all_chunks

    def get_source_stats(self) -> Dict:
        """Get statistics about knowledge sources."""
        stats = {
            "total_sources": len(self.sources),
            "by_type": {},
            "by_priority": {},
            "cached": 0,
            "total_size_mb": 0,
        }

        for source_id, source in self.sources.items():
            # Count by type
            stats["by_type"][source.source_type] = (
                stats["by_type"].get(source.source_type, 0) + 1
            )

            # Count by priority
            stats["by_priority"][source.priority] = (
                stats["by_priority"].get(source.priority, 0) + 1
            )

            # Check if cached
            if self.is_text_cached(source_id):
                stats["cached"] += 1

            # Sum size
            stats["total_size_mb"] += source.size_bytes / (1024 * 1024)

        stats["total_size_mb"] = round(stats["total_size_mb"], 2)

        return stats


def main():
    """Test the knowledge text extractor."""
    extractor = KnowledgeTextExtractor()

    # Print statistics
    stats = extractor.get_source_stats()
    logger.info("Knowledge Sources Statistics:")
    logger.info(f"  Total sources: {stats['total_sources']}")
    logger.info(f"  By type: {stats['by_type']}")
    logger.info(f"  By priority: {stats['by_priority']}")
    logger.info(f"  Cached: {stats['cached']}")
    logger.info(f"  Total size: {stats['total_size_mb']} MB")

    # List sources
    logger.info("\nKnowledge Sources:")
    for source_id, source in extractor.sources.items():
        cached = "✓" if extractor.is_text_cached(source_id) else "○"
        logger.info(
            f"  [{cached}] {source.title} ({source.author}) - {source.priority}"
        )


if __name__ == "__main__":
    main()
