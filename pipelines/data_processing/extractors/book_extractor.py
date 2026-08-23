import os
import tempfile

import ebooklib
import fitz
from bs4 import BeautifulSoup
from ebooklib import epub

from .s3_streamer import S3Streamer


class BookExtractor:
    """Downloads EPUB/PDF books from S3, extracts text, and chunks into knowledge blocks."""

    def __init__(self, streamer: S3Streamer):
        self.streamer = streamer
        self.base_prefix = "archive/local_books_import/"

        # We already installed these via uv
        self.has_pdf = True
        self.has_epub = True

    def chunk_text(self, text, chunk_size=500):
        """Splits text into word chunks."""
        words = text.split()
        for i in range(0, len(words), chunk_size):
            yield " ".join(words[i:i+chunk_size])

    def extract_pdf(self, filepath):
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
        except Exception:
            pass
        return text

    def extract_epub(self, filepath):
        text = ""
        try:
            book = epub.read_epub(filepath)
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_body_content(), "html.parser")
                    text += soup.get_text() + "\n"
        except Exception:
            pass
        return text

    def extract_all(self):
        """Yields chunked knowledge records from all books in S3."""
        if not self.has_pdf or not self.has_epub:
            return

        all_files = list(self.streamer.list_files(self.base_prefix))
        books = [f for f in all_files if f.endswith((".pdf", ".epub"))]

        for file_key in books:
            book_name = os.path.basename(file_key)

            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                temp_path = tmp.name

            try:
                self.streamer.download_to_file(file_key, temp_path)

                text = ""
                if file_key.endswith(".pdf"):
                    text = self.extract_pdf(temp_path)
                elif file_key.endswith(".epub"):
                    text = self.extract_epub(temp_path)

                # Chunk and yield
                for i, chunk in enumerate(self.chunk_text(text)):
                    if len(chunk.strip()) < 50:
                        continue # Skip very small/empty chunks

                    yield {
                        "raw_data": {"text": chunk},
                        "metadata": {
                            "source_family": "psychology_knowledge",
                            "category": "therapeutic_books",
                            "book_name": book_name,
                            "chunk_index": i
                        }
                    }

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
