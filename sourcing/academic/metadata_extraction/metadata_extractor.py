"""
Metadata extraction module for academic sourcing.

Extracts structured metadata from documents (PDF, text, HTML, EPUB)
for use in the academic sourcing pipeline.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Common patterns for metadata extraction
_DOI_PATTERN = re.compile(r"10\.\d{4,}/[^\s,;\"')\]]+")
_ISBN10_PATTERN = re.compile(r"\b(?:\d[\-\s]?){9}[\dXx]\b")
_ISBN13_PATTERN = re.compile(r"\b97[89](?:[\-\s]?\d){10}\b")
_YEAR_PATTERN = re.compile(r"\b((?:19|20)\d{2})\b")
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


@dataclass
class ExtractedMetadata:
    """Structured metadata extracted from a document."""

    title: str | None = None
    authors: list[str] = field(default_factory=list)
    doi: str | None = None
    isbn: str | None = None
    publication_year: int | None = None
    publisher: str | None = None
    abstract: str | None = None
    keywords: list[str] = field(default_factory=list)
    source_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "authors": self.authors,
            "doi": self.doi,
            "isbn": self.isbn,
            "publication_year": self.publication_year,
            "publisher": self.publisher,
            "abstract": self.abstract,
            "keywords": self.keywords,
            "source_format": self.source_format,
        }


class MetadataExtractor:
    """Extract structured metadata from documents.

    Supports plain text, HTML, and PDF (via text extraction) inputs.
    Uses regex-based heuristics to identify DOI, ISBN, authors, title, etc.
    """

    def extract(self, file_path: str) -> ExtractedMetadata:
        """Extract metadata from a file on disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix in (".pdf",):
            text = self._extract_pdf_text(path)
        elif suffix in (".html", ".htm"):
            text = self._extract_html_text(path)
        elif suffix in (".txt", ".md", ".rst"):
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            text = path.read_text(encoding="utf-8", errors="replace")

        return self.extract_from_text(text, source_format=suffix.lstrip("."))

    def extract_from_text(self, text: str, source_format: str | None = None) -> ExtractedMetadata:
        """Extract metadata from raw text content."""
        metadata = ExtractedMetadata(source_format=source_format)

        # DOI
        if match := _DOI_PATTERN.search(text):
            metadata.doi = match.group(0)

        # ISBN (prefer 13-digit)
        if (match := _ISBN13_PATTERN.search(text)) or (match := _ISBN10_PATTERN.search(text)):
            metadata.isbn = match.group(0).replace("-", "").replace(" ", "")

        # Publication year — first occurrence in text
        year_matches = [int(m) for m in _YEAR_PATTERN.findall(text)]
        if year_matches:
            metadata.publication_year = year_matches[0]

        # Title — first non-empty line that looks like a title
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and len(stripped) < 300 and not stripped.startswith(("http", "DOI", "ISBN", "©")):
                metadata.title = stripped
                break

        # Authors — lines containing email addresses or "by" prefix
        for line in text.splitlines()[:50]:
            stripped = line.strip()
            if stripped.lower().startswith("by ") and len(stripped) < 200:
                author_part = stripped[3:]
                metadata.authors = [a.strip() for a in re.split(r"[,;]| and ", author_part) if a.strip()]
                break
            if _EMAIL_PATTERN.search(stripped):
                # Line likely contains author contact info
                names = re.findall(r"([A-Z][a-z]+\s+[A-Z][a-z]+)", stripped)
                if names:
                    metadata.authors = names
                    break

        # Keywords — lines starting with "Keywords:" or "Keywords:"
        keyword_match = re.search(r"Keywords?:\s*(.+)", text, re.IGNORECASE)
        if keyword_match:
            kw_text = keyword_match.group(1).split("\n")[0]
            metadata.keywords = [k.strip() for k in re.split(r"[;,]", kw_text) if k.strip()]

        # Abstract — text between "Abstract" and the next section
        abstract_match = re.search(r"Abstract\s*[:\n]\s*(.+?)(?:\n\n|\nIntroduction|\n1\.\s|\Z)", text, re.IGNORECASE | re.DOTALL)
        if abstract_match:
            metadata.abstract = abstract_match.group(1).strip()[:2000]

        # Publisher — common publisher names
        publisher_match = re.search(r"(?:Published by|Publisher):?\s*(.+)", text, re.IGNORECASE)
        if publisher_match:
            metadata.publisher = publisher_match.group(1).split("\n")[0].strip()[:200]

        return metadata

    def _extract_pdf_text(self, path: Path) -> str:
        """Extract text from PDF. Uses PyMuPDF if available, otherwise pdfminer."""
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(path))
            text = "\n".join(str(page.get_text()) for page in doc)
            doc.close()
            return text
        except ImportError:
            pass
        try:
            from pdfminer.high_level import extract_text

            return extract_text(str(path))
        except ImportError:
            logger.warning("No PDF library available (install PyMuPDF or pdfminer.six)")
            return ""

    def _extract_html_text(self, path: Path) -> str:
        """Extract text from HTML by stripping tags."""
        import html.parser

        class _Stripper(html.parser.HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self._text: list[str] = []

            def handle_data(self, data: str) -> None:
                self._text.append(data)

            def get_text(self) -> str:
                return "".join(self._text)

        raw = path.read_text(encoding="utf-8", errors="replace")
        stripper = _Stripper()
        stripper.feed(raw)
        return stripper.get_text()
