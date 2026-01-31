#!/usr/bin/env python3
"""
Priority 4: PDF Books to Training Data Converter
===============================================

Converts authoritative mental health/therapy PDF books into structured training data
for the Pixelated Empathy AI system. Extracts knowledge and converts to conversation format.

Usage:
    python pdf_to_training_data.py --input_dir /path/to/pdfs --output_file priority_4_books.jsonl
    python pdf_to_training_data.py --single_pdf book.pdf --output_file book_training.jsonl

Features:
- PDF text extraction with multiple fallback methods
- Intelligent chunking by chapters, sections, concepts
- Conversion to therapeutic conversation format
- Quality scoring and metadata preservation
- Support for multiple PDF formats and layouts
"""

import argparse
import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# PDF processing libraries
try:
    import fitz  # pymupdf
except ImportError:
    fitz = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
except ImportError:
    pdfminer_extract = None

# NLP libraries for text processing
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import sent_tokenize, word_tokenize
except ImportError:
    nltk = None

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Extract and process text from PDF files."""

    def __init__(self):
        self.extraction_methods = []
        if fitz:
            self.extraction_methods.append(self._extract_with_pymupdf)
        if pdfplumber:
            self.extraction_methods.append(self._extract_with_pdfplumber)
        if pdfminer_extract:
            self.extraction_methods.append(self._extract_with_pdfminer)

        if not self.extraction_methods:
            raise ImportError(
                "No PDF extraction libraries available. Install pymupdf, pdfplumber, or pdfminer"
            )

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF or Text file using multiple fallback methods."""
        # Simple text file detection
        if pdf_path.lower().endswith((".txt", ".md")):
            try:
                with open(pdf_path, "r", encoding="utf-8") as f:
                    text = f.read()
                if text and len(text.strip()) > 100:
                    logger.info(
                        f"Successfully read {len(text)} characters from text file {pdf_path}"
                    )
                    return text
            except Exception as e:
                logger.warning(f"Failed to read text file {pdf_path}: {e}")

        # Try as PDF even if it doesn't have .pdf extension (or vice versa)
        for method in self.extraction_methods:
            try:
                text = method(pdf_path)
                if text and len(text.strip()) > 100:  # Minimum viable text
                    logger.info(
                        f"Successfully extracted {len(text)} characters from {pdf_path}"
                    )
                    return text
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed for {pdf_path}: {e}")
                continue

        # Fallback: try opening as a plain text file regardless of extension
        try:
            with open(pdf_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if text and len(text.strip()) > 100:
                logger.info(
                    f"Fallback: read {len(text)} characters as plain text from {pdf_path}"
                )
                return text
        except Exception as e:
            logger.warning(f"Final fallback failed for {pdf_path}: {e}")

        raise RuntimeError(f"All extraction methods failed for {pdf_path}")

    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF."""
        doc = fitz.open(pdf_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text

    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                if page_text := page.extract_text():
                    text += page_text + "\n"
        return text

    def _extract_with_pdfminer(self, pdf_path: str) -> str:
        """Extract text using pdfminer."""
        return pdfminer_extract(pdf_path)


class TextChunker:
    """Intelligently chunk extracted text into meaningful sections."""

    def __init__(self):
        # Common chapter/section patterns
        self.chapter_patterns = [
            r"^Chapter\s+\d+",
            r"^CHAPTER\s+\d+",
            r"^\d+\.\s+[A-Z]",
            r"^Part\s+\d+",
            r"^Section\s+\d+",
        ]

        # Therapeutic concept patterns
        self.concept_patterns = [
            r"cognitive\s+behavioral\s+therapy",
            r"dialectical\s+behavior\s+therapy",
            r"acceptance\s+and\s+commitment\s+therapy",
            r"mindfulness",
            r"therapeutic\s+alliance",
            r"transference",
            r"countertransference",
            r"attachment\s+theory",
            r"trauma\s+informed",
        ]

    def chunk_by_structure(self, text: str) -> List[Dict[str, Any]]:
        """Chunk text by structural elements (chapters, sections)."""
        chunks = []
        lines = text.split("\n")
        current_chunk = []
        current_title = "Introduction"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check if this line is a chapter/section header
            is_header = False
            for pattern in self.chapter_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    # Save previous chunk
                    if current_chunk:
                        chunks.append(
                            {
                                "title": current_title,
                                "content": "\n".join(current_chunk),
                                "type": "structural_section",
                            }
                        )

                    # Start new chunk
                    current_title = line
                    current_chunk = []
                    is_header = True
                    break

            if not is_header:
                current_chunk.append(line)

        # Add final chunk
        if current_chunk:
            chunks.append(
                {
                    "title": current_title,
                    "content": "\n".join(current_chunk),
                    "type": "structural_section",
                }
            )

        return chunks

    def chunk_by_concepts(
        self, text: str, max_chunk_size: int = 2000
    ) -> List[Dict[str, Any]]:
        """Chunk text by therapeutic concepts and ideas."""
        chunks = []
        sentences = sent_tokenize(text) if nltk else text.split(". ")

        current_chunk = []
        current_size = 0
        current_concepts = set()

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Identify concepts in this sentence
            sentence_concepts = set()
            for pattern in self.concept_patterns:
                if re.search(pattern, sentence, re.IGNORECASE):
                    concept = re.search(pattern, sentence, re.IGNORECASE).group()
                    sentence_concepts.add(concept.lower())

            # Check if we should start a new chunk
            sentence_size = len(sentence)
            if (current_size + sentence_size > max_chunk_size and current_chunk) or (
                sentence_concepts
                and current_concepts
                and not sentence_concepts.intersection(current_concepts)
            ):
                # Save current chunk
                chunks.append(
                    {
                        "content": " ".join(current_chunk),
                        "concepts": list(current_concepts),
                        "type": "concept_section",
                        "size": current_size,
                    }
                )

                # Start new chunk
                current_chunk = [sentence]
                current_size = sentence_size
                current_concepts = sentence_concepts
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                current_concepts.update(sentence_concepts)

        # Add final chunk
        if current_chunk:
            chunks.append(
                {
                    "content": " ".join(current_chunk),
                    "concepts": list(current_concepts),
                    "type": "concept_section",
                    "size": current_size,
                }
            )

        return chunks


class TrainingDataConverter:
    """Convert text chunks into training conversation format."""

    def __init__(self):
        self.conversation_templates = {
            "explanation": [
                "Can you explain {concept}?",
                "What is {concept} and how is it used in therapy?",
                "How would you describe {concept} to a client?",
                "What are the key principles of {concept}?",
            ],
            "application": [
                "How would you apply {concept} in a therapy session?",
                "Can you give an example of {concept} in practice?",
                "When would you use {concept} with a client?",
                "What are some techniques related to {concept}?",
            ],
            "case_study": [
                "How might {concept} help with {issue}?",
                "Can you walk through using {concept} for {issue}?",
                "What would {concept} look like when treating {issue}?",
            ],
        }

        self.common_issues = [
            "depression",
            "anxiety",
            "trauma",
            "relationship issues",
            "grief",
            "addiction",
            "personality disorders",
            "eating disorders",
        ]

    def chunk_to_conversations(
        self, chunk: Dict[str, Any], book_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Convert a text chunk into multiple training conversations."""
        conversations = []
        content = chunk["content"]

        # Extract key concepts from the chunk
        concepts = chunk.get("concepts") or self._extract_key_terms(content)

        # Generate different types of conversations
        for concept in concepts[:3]:  # Limit to top 3 concepts per chunk
            # Explanation conversation
            conversations.extend(
                self._create_explanation_conversation(content, concept, book_metadata)
            )

            # Application conversation
            conversations.extend(
                self._create_application_conversation(content, concept, book_metadata)
            )

            # Case study conversation (if applicable)
            if len(content) > 500:  # Only for substantial content
                conversations.extend(
                    self._create_case_study_conversation(
                        content, concept, book_metadata
                    )
                )

        return conversations

    def _create_explanation_conversation(
        self, content: str, concept: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create explanation-focused conversation."""
        # Extract relevant explanation from content
        explanation = self._extract_relevant_text(content, concept, max_length=800)

        if not explanation:
            return []

        question_template = self.conversation_templates["explanation"][0]
        question = question_template.format(concept=concept)

        conversation = {
            "id": str(uuid.uuid4()),
            "source_priority": "4",
            "source_dataset": f"book_{metadata.get('title', 'unknown')}",
            "conversation_type": "therapeutic_knowledge",
            "scenario_category": "theoretical_explanation",
            "quality_score": self._calculate_quality_score(explanation),
            "conversation": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": explanation,
                    "reasoning": f"Based on authoritative knowledge from {metadata.get('author', 'therapeutic literature')}",
                },
            ],
            "metadata": {
                "difficulty_level": 2,
                "therapeutic_concept": concept,
                "source_book": metadata.get("title"),
                "source_author": metadata.get("author"),
                "chunk_type": "explanation",
                "split": "train",  # Will be updated during final processing
            },
        }

        return [conversation]

    def _create_application_conversation(
        self, content: str, concept: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create application-focused conversation."""
        application_text = self._extract_application_text(content, concept)

        if not application_text:
            return []

        question = f"How would you apply {concept} in a therapy session?"

        conversation = {
            "id": str(uuid.uuid4()),
            "source_priority": "4",
            "source_dataset": f"book_{metadata.get('title', 'unknown')}",
            "conversation_type": "therapeutic_application",
            "scenario_category": "practical_application",
            "quality_score": self._calculate_quality_score(application_text),
            "conversation": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": application_text,
                    "reasoning": f"Practical application based on {metadata.get('author', 'therapeutic literature')}'s approach",
                },
            ],
            "metadata": {
                "difficulty_level": 3,
                "therapeutic_concept": concept,
                "source_book": metadata.get("title"),
                "source_author": metadata.get("author"),
                "chunk_type": "application",
                "split": "train",
            },
        }

        return [conversation]

    def _create_case_study_conversation(
        self, content: str, concept: str, metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Create case study conversation."""
        # Select a relevant issue for the case study
        issue = self._select_relevant_issue(content)
        case_study_text = self._create_case_study_text(content, concept, issue)

        if not case_study_text:
            return []

        question = f"How might {concept} help with {issue}?"

        conversation = {
            "id": str(uuid.uuid4()),
            "source_priority": "4",
            "source_dataset": f"book_{metadata.get('title', 'unknown')}",
            "conversation_type": "therapeutic_case_study",
            "scenario_category": "case_application",
            "quality_score": self._calculate_quality_score(case_study_text),
            "conversation": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": case_study_text,
                    "reasoning": f"Case study approach based on {concept} principles from {metadata.get('author', 'therapeutic literature')}",
                },
            ],
            "metadata": {
                "difficulty_level": 4,
                "therapeutic_concept": concept,
                "client_issue": issue,
                "source_book": metadata.get("title"),
                "source_author": metadata.get("author"),
                "chunk_type": "case_study",
                "split": "train",
            },
        }

        return [conversation]

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key therapeutic terms from text."""
        key_terms = []
        for pattern in TextChunker().concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            key_terms.extend([match.lower() for match in matches])

        # Add some general terms if no specific concepts found
        if not key_terms:
            general_terms = [
                "therapy",
                "therapeutic relationship",
                "client care",
                "mental health",
            ]
            key_terms.extend(
                term for term in general_terms if term.lower() in text.lower()
            )

        return list(set(key_terms))[:5]  # Return up to 5 unique terms

    def _extract_relevant_text(
        self, content: str, concept: str, max_length: int = 800
    ) -> str:
        """Extract text most relevant to the concept."""
        sentences = sent_tokenize(content) if nltk else content.split(". ")
        relevant_sentences = []

        # Find sentences containing the concept
        relevant_sentences.extend(
            sentence.strip()
            for sentence in sentences
            if concept.lower() in sentence.lower()
        )
        # If no direct matches, find contextually relevant sentences
        if not relevant_sentences:
            concept_words = concept.lower().split()
            relevant_sentences.extend(
                sentence.strip()
                for sentence in sentences
                if any(word in sentence.lower() for word in concept_words)
            )

        # Combine sentences up to max_length
        result = ""
        for sentence in relevant_sentences:
            if len(result + sentence) <= max_length:
                result = f"{result}{sentence} "
            else:
                break

        return result.strip()

    def _extract_application_text(self, content: str, concept: str) -> str:
        """Extract text about practical application of the concept."""
        # Look for application-oriented sentences
        application_keywords = [
            "apply",
            "use",
            "practice",
            "technique",
            "method",
            "approach",
            "session",
            "client",
        ]
        sentences = sent_tokenize(content) if nltk else content.split(". ")

        relevant_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if concept.lower() in sentence_lower and any(
                keyword in sentence_lower for keyword in application_keywords
            ):
                relevant_sentences.append(sentence.strip())

        return " ".join(relevant_sentences[:3])  # Limit to 3 sentences

    def _select_relevant_issue(self, content: str) -> str:
        """Select the most relevant mental health issue mentioned in content."""
        content_lower = content.lower()
        return next(
            (issue for issue in self.common_issues if issue in content_lower),
            "general mental health concerns",
        )

    def _create_case_study_text(self, content: str, concept: str, issue: str) -> str:
        """Create case study text combining concept and issue."""
        relevant_text = self._extract_relevant_text(content, concept, max_length=600)
        if relevant_text:
            return f"When working with {issue}, {concept} can be particularly helpful. {relevant_text} This approach allows therapists to address the specific challenges associated with {issue} while building on the client's strengths and resources."
        return ""

    def _calculate_quality_score(self, text: str) -> float:
        """Calculate quality score for the training sample."""
        if not text:
            return 0.0

        score = 0.5  # Base score

        # Length bonus (optimal range)
        length = len(text)
        if 200 <= length <= 1000:
            score += 0.2
        elif 100 <= length < 200 or 1000 < length <= 1500:
            score += 0.1

        # Professional language bonus
        professional_terms = [
            "therapy",
            "therapeutic",
            "client",
            "treatment",
            "intervention",
            "assessment",
        ]
        term_count = len([term for term in professional_terms if term in text.lower()])
        score += min(term_count * 0.05, 0.2)

        # Structure bonus (sentences, punctuation)
        sentence_count = len(sent_tokenize(text)) if nltk else text.count(".")
        if 3 <= sentence_count <= 10:
            score += 0.1

        return min(score, 1.0)

    def process_text_file(self, file_path, output_handle):
        """Process a text/markdown file directly."""
        text = self.extract_text(file_path)
        enhanced_text = enhance_therapy_text(text)

        # Use TextChunker
        chunker = TextChunker()
        concept_chunks = chunker.chunk_by_concepts(enhanced_text)

        converter = TrainingDataConverter()
        all_conversations = []

        book_metadata = {
            "title": Path(file_path).stem,
            "author": "Unknown",
            "year": datetime.now().year,
        }

        for chunk in concept_chunks:
            conversations = converter.chunk_to_conversations(chunk, book_metadata)
            all_conversations.extend(conversations)

        return all_conversations


def process_pdf_file(
    pdf_path: str, output_file: str, book_metadata: Dict[str, Any] = None
):
    """Process a single PDF file and convert to training data."""
    if not book_metadata:
        book_metadata = {
            "title": Path(pdf_path).stem,
            "author": "Unknown",
            "year": datetime.now().year,
        }

    logger.info(f"Processing PDF: {pdf_path}")

    # Extract text
    processor = PDFProcessor()
    try:
        text = processor.extract_text(pdf_path)
    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        return

    # Chunk text
    chunker = TextChunker()
    structural_chunks = chunker.chunk_by_structure(text)
    concept_chunks = chunker.chunk_by_concepts(text)

    all_chunks = structural_chunks + concept_chunks
    logger.info(f"Created {len(all_chunks)} chunks from {pdf_path}")

    # Convert to training data
    converter = TrainingDataConverter()
    all_conversations = []

    for chunk in all_chunks:
        conversations = converter.chunk_to_conversations(chunk, book_metadata)
        all_conversations.extend(conversations)

    logger.info(
        f"Generated {len(all_conversations)} training conversations from {pdf_path}"
    )

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        for conversation in all_conversations:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    logger.info(f"Saved training data to {output_file}")

    # Generate summary
    summary = {
        "source_file": pdf_path,
        "book_metadata": book_metadata,
        "processing_date": datetime.now().isoformat(),
        "total_chunks": len(all_chunks),
        "total_conversations": len(all_conversations),
        "average_quality_score": sum(c["quality_score"] for c in all_conversations)
        / len(all_conversations)
        if all_conversations
        else 0,
        "conversation_types": {
            "therapeutic_knowledge": len(
                [
                    c
                    for c in all_conversations
                    if c["conversation_type"] == "therapeutic_knowledge"
                ]
            ),
            "therapeutic_application": len(
                [
                    c
                    for c in all_conversations
                    if c["conversation_type"] == "therapeutic_application"
                ]
            ),
            "therapeutic_case_study": len(
                [
                    c
                    for c in all_conversations
                    if c["conversation_type"] == "therapeutic_case_study"
                ]
            ),
        },
    }

    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Processing complete. Summary saved to {summary_file}")


def main():
    parser = argparse.ArgumentParser(description="Convert PDF books to training data")
    parser.add_argument("--input_dir", type=str, help="Directory containing PDF files")
    parser.add_argument("--single_pdf", type=str, help="Single PDF file to process")
    parser.add_argument(
        "--output_file", type=str, required=True, help="Output JSONL file"
    )
    parser.add_argument("--book_title", type=str, help="Book title (for single PDF)")
    parser.add_argument("--book_author", type=str, help="Book author (for single PDF)")
    parser.add_argument(
        "--book_year", type=int, help="Publication year (for single PDF)"
    )

    args = parser.parse_args()

    if not args.input_dir and not args.single_pdf:
        parser.error("Either --input_dir or --single_pdf must be specified")

    if args.single_pdf:
        # Process single PDF
        book_metadata = {
            "title": args.book_title or Path(args.single_pdf).stem,
            "author": args.book_author or "Unknown",
            "year": args.book_year or datetime.now().year,
        }
        process_pdf_file(args.single_pdf, args.output_file, book_metadata)

    elif args.input_dir:
        # Process directory of PDFs and Text files
        pattern = "*.*"  # All files
        files = [
            f
            for f in Path(args.input_dir).glob(pattern)
            if f.suffix.lower() in (".pdf", ".txt", ".md")
        ]

        if not files:
            logger.error(f"No valid files found in {args.input_dir}")
            return

        all_conversations = []
        converter = TrainingDataConverter()

        for f in files:
            logger.info(f"Processing {f.name}...")
            if f.suffix.lower() == ".pdf":
                temp_output = f"temp_{f.stem}.jsonl"
                process_pdf_file(str(f), temp_output)
                # Read back
                if os.path.exists(temp_output):
                    with open(temp_output, "r", encoding="utf-8") as tf:
                        all_conversations.extend(json.loads(line) for line in tf)
                    os.remove(temp_output)
            else:
                # Text file
                convs = converter.process_text_file(str(f), None)
                all_conversations.extend(convs)

        # Save combined output
        with open(args.output_file, "w", encoding="utf-8") as f:
            for conversation in all_conversations:
                f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

        logger.info(
            f"Combined {len(all_conversations)} conversations from {len(files)} files"
        )


def enhance_therapy_text(text):
    """Clean and enhance extracted text for therapy books"""
    # Remove page headers/footers
    cleaned_text = text.replace("\f", "\n")  # Form feeds to newlines
    # Fix hyphenated words across lines
    cleaned_text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", cleaned_text)
    # Preserve therapeutic terminology
    # Clean up citation formatting
    return cleaned_text


if __name__ == "__main__":
    main()
