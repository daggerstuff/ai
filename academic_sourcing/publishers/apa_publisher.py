"""
APA Publisher Integration for Academic Sourcing Engine

This module provides integration with the American Psychological Association (APA)
for sourcing psychology and therapy books for AI training data expansion.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APAPublisher(BasePublisher):
    """Integration with American Psychological Association (APA) for book sourcing"""

    def __init__(self):
        """
        Initialize APA publisher integration
        """
        super().__init__(
            name="American Psychological Association",
            api_base_url="https://api.apa.org/v2",
            requires_auth=True,
        )

        # APA-specific therapeutic topics mapping
        self.therapeutic_topics_map = {
            "cbt": [
                "cognitive behavioral therapy",
                "cognitive therapy",
                "behavioral therapy",
            ],
            "dbt": ["dialectical behavior therapy", "dialectical therapy"],
            "trauma": ["trauma", "ptsd", "post-traumatic stress", "childhood trauma"],
            "anxiety": [
                "anxiety",
                "generalized anxiety",
                "social anxiety",
                "panic disorder",
            ],
            "depression": ["depression", "major depressive disorder", "dysthymia"],
            "personality": [
                "personality disorders",
                "borderline personality",
                "narcissistic personality",
            ],
            "addiction": [
                "addiction",
                "substance abuse",
                "alcoholism",
                "drug dependence",
            ],
            "child": [
                "child psychology",
                "adolescent psychology",
                "developmental psychology",
            ],
            "family": ["family therapy", "couples therapy", "marriage counseling"],
            "cultural": [
                "cultural competence",
                "multicultural psychology",
                "diversity",
            ],
            "ethics": ["ethics", "professional ethics", "confidentiality"],
            "assessment": ["assessment", "psychological testing", "diagnostic tools"],
            "neuropsychology": [
                "neuropsychology",
                "brain injury",
                "cognitive rehabilitation",
            ],
            "mindfulness": ["mindfulness", "meditation", "stress reduction"],
        }

    def _authenticate(self, api_key: Optional[str] = None, **kwargs) -> bool:
        """
        Authenticate with APA API

        Args:
            api_key: APA API key
            **kwargs: Additional authentication parameters

        Returns:
            bool: True if authentication succeeded, False otherwise
        """
        if not api_key:
            logger.error("API key is required for APA authentication")
            return False

        try:
            # APA uses OAuth2 with API key
            auth_response = self._make_request(
                method="POST",
                endpoint="/auth/token",
                data={
                    "grant_type": "api_key",
                    "api_key": api_key,
                    "client_id": kwargs.get("client_id", "pixelated-empathy"),
                },
            )

            if auth_response and "access_token" in auth_response:
                self._auth_token = auth_response["access_token"]
                return True
            else:
                logger.error("APA authentication failed: Invalid response")
                return False

        except Exception as e:
            logger.error(f"APA authentication error: {e}")
            return False

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """
        Search for books in APA's catalog

        Args:
            query: Search query (title, author, keywords, etc.)
            year_range: Tuple of (start_year, end_year) to filter by publication year
            therapeutic_topics: List of therapeutic topics to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of BookMetadata objects matching the search criteria
        """
        if not self._auth_token:
            logger.error("Not authenticated with APA. Call authenticate() first.")
            return []

        # Build search parameters
        params = {"q": query, "limit": limit, "offset": offset, "format": "json"}

        # Add year range filter
        if year_range:
            params["publication_year_from"] = year_range[0]
            params["publication_year_to"] = year_range[1]

        # Add therapeutic topics filter
        if therapeutic_topics:
            topic_queries = []
            for topic in therapeutic_topics:
                if topic.lower() in self.therapeutic_topics_map:
                    topic_queries.extend(self.therapeutic_topics_map[topic.lower()])
                else:
                    topic_queries.append(topic)

            if topic_queries:
                params["q"] += " AND (" + " OR ".join(topic_queries) + ")"

        try:
            # Search for books
            search_response = self._make_request(
                method="GET", endpoint="/books/search", params=params
            )

            if not search_response or "results" not in search_response:
                logger.warning("No results or invalid response from APA search")
                return []

            # Process results
            book_metadata_list = []
            for book_data in search_response["results"]:
                metadata = self._parse_book_metadata(book_data)
                if metadata:
                    # Assess therapeutic relevance
                    self.assess_therapeutic_relevance(metadata)
                    book_metadata_list.append(metadata)

            return book_metadata_list

        except Exception as e:
            logger.error(f"Error searching APA books: {e}")
            return []

    def get_book_metadata(self, identifier: str) -> Optional[BookMetadata]:
        """
        Get metadata for a specific book from APA

        Args:
            identifier: Book identifier (ISBN, DOI, or APA ID)

        Returns:
            BookMetadata object if found, None otherwise
        """
        if not self._auth_token:
            logger.error("Not authenticated with APA. Call authenticate() first.")
            return None

        try:
            # Determine identifier type
            if identifier.startswith("10."):  # DOI
                endpoint = f"/books/doi/{identifier}"
            elif len(identifier) == 10 or len(identifier) == 13:  # ISBN
                endpoint = f"/books/isbn/{identifier}"
            else:  # Assume APA ID
                endpoint = f"/books/{identifier}"

            # Get book metadata
            book_response = self._make_request(method="GET", endpoint=endpoint)

            if not book_response:
                logger.warning(f"No book found with identifier: {identifier}")
                return None

            metadata = self._parse_book_metadata(book_response)
            if metadata:
                self.assess_therapeutic_relevance(metadata)
                return metadata

            return None

        except Exception as e:
            logger.error(f"Error getting APA book metadata: {e}")
            return None

    def get_book_content(
        self, identifier: str, format: BookFormat = BookFormat.PLAIN_TEXT
    ) -> Optional[BookContent]:
        """
        Get content for a specific book from APA

        Note: APA typically provides content in PDF format, which we'll convert to plain text

        Args:
            identifier: Book identifier (ISBN, DOI, or APA ID)
            format: Desired format for the book content

        Returns:
            BookContent object if found and accessible, None otherwise
        """
        if not self._auth_token:
            logger.error("Not authenticated with APA. Call authenticate() first.")
            return None

        try:
            # First get metadata
            metadata = self.get_book_metadata(identifier)
            if not metadata:
                return None

            # Determine endpoint based on identifier type
            if identifier.startswith("10."):  # DOI
                endpoint = f"/books/doi/{identifier}/content"
            elif len(identifier) == 10 or len(identifier) == 13:  # ISBN
                endpoint = f"/books/isbn/{identifier}/content"
            else:  # Assume APA ID
                endpoint = f"/books/{identifier}/content"

            # Get book content
            content_response = self._make_request(
                method="GET",
                endpoint=endpoint,
                params={"format": "pdf"},  # APA typically provides PDFs
            )

            if not content_response or "content_url" not in content_response:
                logger.warning(f"No content available for book: {identifier}")
                return None

            # Download the content
            content_url = content_response["content_url"]
            content_data = self._download_content(content_url)

            if not content_data:
                return None

            # Create BookContent object
            book_content = BookContent(
                metadata=metadata,
                content="",  # Will be populated after PDF conversion
                format=BookFormat.PDF,
                chapter_contents=[],
            )

            # Convert PDF to text (this would be implemented in a separate module)
            # For now, we'll just store the PDF data and return
            # In a real implementation, we would use a PDF processing library
            book_content.content = f"PDF_CONTENT_FOR_{identifier}"

            return book_content

        except Exception as e:
            logger.error(f"Error getting APA book content: {e}")
            return None

    def get_chapter_content(self, identifier: str, chapter_id: str) -> Optional[str]:
        """
        Get content for a specific chapter of a book from APA

        Args:
            identifier: Book identifier (ISBN, DOI, or APA ID)
            chapter_id: Chapter identifier

        Returns:
            Chapter content as string if found, None otherwise
        """
        if not self._auth_token:
            logger.error("Not authenticated with APA. Call authenticate() first.")
            return None

        try:
            # Determine endpoint based on identifier type
            if identifier.startswith("10."):  # DOI
                endpoint = f"/books/doi/{identifier}/chapters/{chapter_id}"
            elif len(identifier) == 10 or len(identifier) == 13:  # ISBN
                endpoint = f"/books/isbn/{identifier}/chapters/{chapter_id}"
            else:  # Assume APA ID
                endpoint = f"/books/{identifier}/chapters/{chapter_id}"

            # Get chapter content
            chapter_response = self._make_request(method="GET", endpoint=endpoint)

            if not chapter_response or "content" not in chapter_response:
                logger.warning(
                    f"No chapter content found for {identifier} chapter {chapter_id}"
                )
                return None

            return chapter_response["content"]

        except Exception as e:
            logger.error(f"Error getting APA chapter content: {e}")
            return None

    def _parse_book_metadata(self, book_data: Dict[str, Any]) -> Optional[BookMetadata]:
        """
        Parse APA book data into BookMetadata object

        Args:
            book_data: Raw book data from APA API

        Returns:
            BookMetadata object if parsing succeeded, None otherwise
        """
        try:
            # Extract authors
            authors = []
            if "authors" in book_data:
                for author in book_data["authors"]:
                    if isinstance(author, str):
                        authors.append(author)
                    elif isinstance(author, dict) and "name" in author:
                        authors.append(author["name"])

            # Extract keywords
            keywords = []
            if "keywords" in book_data:
                keywords = [kw.strip() for kw in book_data["keywords"] if kw.strip()]

            # Extract chapters if available
            chapters = None
            if "chapters" in book_data:
                chapters = []
                for chapter in book_data["chapters"]:
                    if isinstance(chapter, dict):
                        chapters.append(
                            {
                                "title": chapter.get("title", ""),
                                "page_range": chapter.get("page_range", ""),
                                "chapter_id": chapter.get("id", ""),
                            }
                        )

            # Create metadata object
            metadata = BookMetadata(
                title=book_data.get("title", "Unknown Title"),
                authors=authors,
                publisher="American Psychological Association",
                publication_year=book_data.get("publication_year", datetime.now().year),
                isbn=book_data.get("isbn", None),
                doi=book_data.get("doi", None),
                abstract=book_data.get("abstract", None),
                keywords=keywords,
                chapters=chapters,
                page_count=book_data.get("page_count", None),
                language=book_data.get("language", "en"),
                license=book_data.get("license", None),
                copyright_status=book_data.get("copyright_status", None),
                source_publisher="apa",
                raw_metadata=book_data,
            )

            return metadata

        except Exception as e:
            logger.error(f"Error parsing APA book metadata: {e}")
            return None

    def _download_content(self, content_url: str) -> Optional[bytes]:
        """
        Download content from a URL

        Args:
            content_url: URL to download content from

        Returns:
            Content as bytes if successful, None otherwise
        """
        try:
            response = self.session.get(
                content_url, headers=self._get_headers(), timeout=60
            )

            response.raise_for_status()
            return response.content

        except Exception as e:
            logger.error(f"Error downloading content from {content_url}: {e}")
            return None

    def get_therapeutic_topics(self) -> Dict[str, List[str]]:
        """
        Get the mapping of therapeutic topics to APA search terms

        Returns:
            Dictionary mapping therapeutic topics to APA search terms
        """
        return self.therapeutic_topics_map
