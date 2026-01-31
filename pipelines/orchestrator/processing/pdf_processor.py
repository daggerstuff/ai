#!/usr/bin/env python3
"""
PDF Processing for Psychology Textbook Extraction
Extracts knowledge from psychology PDFs and converts it to standard Conversation schema.
"""

import json
import logging

# Import conversation schema
import sys
import uuid
from pathlib import Path
from pathlib import Path as PathType
from typing import Any, Dict, List, Optional

import pypdf
from tqdm import tqdm

pipeline_root = PathType(__file__).parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    # Fallback for different execution environments
    try:
        from ai.pipelines.orchestrator.schemas.conversation_schema import (
            Conversation,
            Message,
        )
    except ImportError:
        # Define minimal classes if imports fail
        from dataclasses import dataclass, field
        from datetime import datetime, timezone

        @dataclass
        class Message:
            role: str
            content: str
            timestamp: str = field(
                default_factory=lambda: datetime.now(timezone.utc).isoformat()
            )
            metadata: dict[str, Any] = field(default_factory=dict)

            def to_dict(self):
                return {"role": self.role, "content": self.content}

        @dataclass
        class Conversation:
            conversation_id: str
            source: str
            messages: List[Message]
            metadata: Dict[str, Any]

            def to_dict(self):
                return {
                    "conversation_id": self.conversation_id,
                    "source": self.source,
                    "messages": [m.to_dict() for m in self.messages],
                    "metadata": self.metadata,
                }


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Processor for extracting knowledge from PDF textbooks."""

    def __init__(self, output_dir: str = "ai/data/processed/knowledge/pdf_extractions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = 2000  # Characters per chunk
        self.overlap = 200  # Overlap between chunks

    def process_pdf(self, pdf_path: str, source_name: Optional[str] = None) -> str:
        """
        Process a single PDF file and save as JSONL.

        Args:
            pdf_path: Path to the PDF file
            source_name: Optional name for the source

        Returns:
            Path to the generated JSONL file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        source_name = source_name or pdf_path.stem
        logger.info(f"📄 Processing PDF: {pdf_path} (Source: {source_name})")

        # 1. Extract text
        full_text = self._extract_text(pdf_path)
        if not full_text:
            logger.warning(f"⚠️ No text extracted from {pdf_path}")
            return ""

        # 2. Chunk text
        chunks = self._chunk_text(full_text)
        logger.info(f"✂️ Created {len(chunks)} chunks from {pdf_path}")

        # 3. Convert to Conversations
        conversations = self._convert_to_conversations(chunks, source_name)

        # 4. Save to JSONL
        output_file = self.output_dir / f"{pdf_path.stem}_extracted.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            for conv in conversations:
                f.write(json.dumps(conv.to_dict()) + "\n")

        logger.info(f"✅ Saved {len(conversations)} conversations to {output_file}")
        return str(output_file)

    def _extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file."""
        text_parts = []
        try:
            with open(pdf_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num in tqdm(range(len(reader.pages)), desc="Extracting pages"):
                    page = reader.pages[page_num]
                    if page_text := page.extract_text():
                        text_parts.append(page_text)
        except Exception as e:
            logger.error(f"❌ Error extracting text from {pdf_path}: {e}")
            return ""

        return "\n\n".join(text_parts)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        # Simple character-based chunking for now
        # In production, we'd use a more sophisticated sentence-aware chunker
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            # Try to expand to the end of a sentence if not at the very end
            if end < len(text):
                # Look for the last period in the last 200 characters of the chunk
                last_period = text.rfind(".", end - 200, end)
                if last_period != -1:
                    end = last_period + 1

            chunks.append(text[start:end].strip())
            start = max(0, end - self.overlap)
            if end == len(text):
                break

        return [c for c in chunks if len(c) > 100]  # Filter out very small chunks

    def _convert_to_conversations(
        self, chunks: List[str], source_name: str
    ) -> List[Conversation]:
        """Convert text chunks into instruction-following conversations."""
        conversations = []
        for i, chunk in enumerate(chunks):
            # Create a synthetic instruction for the chunk
            # For knowledge base, we often use generic prompts like
            # "Explain this section from [Source]" or extract a title
            # from the first line
            lines = chunk.split("\n")
            title_guess = lines[0].strip() if lines else "information"
            if len(title_guess) > 50:
                title_guess = f"{title_guess[:47]}..."

            instruction = (
                f"Explain the therapeutic concepts discussed in the following section "
                f"from '{source_name}':\n\n[Section starts with: {title_guess}]"
            )

            conv = Conversation(
                conversation_id=str(uuid.uuid4()),
                source=f"pdf_extraction_{source_name}",
                messages=[
                    Message(role="user", content=instruction),
                    Message(role="assistant", content=chunk),
                ],
                metadata={
                    "original_source": source_name,
                    "chunk_index": i,
                    "is_knowledge_base": True,
                    "data_type": "psychology_textbook",
                },
            )
            conversations.append(conv)
        return conversations


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Process psychology PDFs into training data."
    )
    parser.add_argument("pdf_paths", nargs="+", help="Paths to PDF files to process")
    parser.add_argument(
        "--output-dir",
        default="ai/data/processed/knowledge/pdf_extractions",
        help="Output directory",
    )

    args = parser.parse_args()

    processor = PDFProcessor(output_dir=args.output_dir)
    for pdf_path in args.pdf_paths:
        try:
            processor.process_pdf(pdf_path)
        except Exception as e:
            logger.error(f"Failed to process {pdf_path}: {e}")


if __name__ == "__main__":
    main()
