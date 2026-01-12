"""
Springer Publisher Integration for Academic Sourcing Engine

Provides integration with Springer Nature for sourcing psychology and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class SpringerPublisher(BasePublisher):
    """Integration with Springer Nature for book sourcing"""

    def __init__(self):
        """Initialize Springer publisher integration"""
        super().__init__(
            name="Springer Nature",
            api_base_url="https://api.springernature.com",
            requires_auth=True,
        )

        # Springer-specific therapeutic topics mapping
        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "cognitive therapy"],
            "dbt": ["dialectical behavior therapy"],
            "trauma": ["trauma", "ptsd", "post-traumatic stress"],
            "anxiety": ["anxiety disorders", "panic disorder", "social anxiety"],
            "depression": ["depression", "major depressive disorder"],
            "personality": ["personality disorders", "borderline personality"],
            "addiction": ["addiction", "substance abuse"],
            "child": ["child psychology", "adolescent psychology"],
            "family": ["family therapy", "couples therapy"],
            "mindfulness": ["mindfulness", "meditation"],
            "neuropsychology": ["neuropsychology", "brain injury"],
            "assessment": ["psychological assessment", "diagnostic tools"],
        }

        logger.info("Initialized Springer Nature publisher integration")

    def authenticate(self, api_key: str) -> bool:
        """
        Authenticate with Springer API

        Args:
            api_key: Springer API key

        Returns:
            True if authentication successful
        """
        self._auth_token = api_key

        # Test authentication
        try:
            test_url = f"{self.api_base_url}/metadata/json"
            params = {"q": "psychology", "p": 1, "api_key": api_key}

            response = self.session.get(test_url, params=params, timeout=10)

            if response.status_code == 200:
                logger.info("✅ Springer authentication successful")
                return True
            else:
                logger.error(f"Springer authentication failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Springer authentication error: {e}")
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
        Search for books in Springer's catalog

        Args:
            query: Search query
            year_range: Optional (start_year, end_year) tuple
            therapeutic_topics: Optional list of therapeutic topics
            limit: Maximum results to return
            offset: Pagination offset

        Returns:
            List of BookMetadata objects
        """
        if not self._auth_token:
            logger.error("Springer API key not set. Call authenticate() first.")
            return []

        logger.info(f"🔍 Searching Springer for: '{query}'")

        # Build search query
        search_terms = [query]

        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(f'"{term}"' for term in search_terms[:5])

        # API parameters
        params = {
            "q": combined_query,
            "p": limit,
            "s": offset + 1,  # Springer uses 1-based indexing
            "api_key": self._auth_token,
        }

        # Add year filter if specified
        if year_range:
            start_year, end_year = year_range
            params["q"] += f" year:{start_year}-{end_year}"

        try:
            url = f"{self.api_base_url}/metadata/json"
            response = self.session.get(url, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"Springer API error: {response.status_code}")
                return []

            data = response.json()
            records = data.get("records", [])

            books = []
            for record in records:
                try:
                    metadata = self._parse_springer_record(record, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing Springer record: {e}")
                    continue

            logger.info(f"✅ Found {len(books)} books from Springer")
            return books

        except Exception as e:
            logger.error(f"Springer search error: {e}")
            return []

    def _parse_springer_record(
        self, record: Dict[str, Any], original_query: str
    ) -> Optional[BookMetadata]:
        """Parse a Springer API record into BookMetadata"""
        try:
            # Extract basic info
            title = record.get("title", "Unknown Title")

            # Extract authors
            authors = []
            creators = record.get("creators", [])
            for creator in creators:
                if isinstance(creator, dict):
                    name = creator.get("creator", "")
                elif isinstance(creator, str):
                    name = creator
                else:
                    continue

                if name:
                    authors.append(name)

            # Extract publication info
            publication_name = record.get("publicationName", "Springer")
            pub_date = record.get("publicationDate", "")

            year = 0
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                except (ValueError, IndexError):
                    pass

            # Extract identifiers
            isbn = None
            doi = None

            identifiers = record.get("identifier", [])
            for identifier in identifiers:
                if isinstance(identifier, dict):
                    id_type = identifier.get("type", "").lower()
                    id_value = identifier.get("value", "")

                    if "isbn" in id_type:
                        isbn = id_value
                    elif "doi" in id_type:
                        doi = id_value

            # Extract URL
            url = record.get("url", [])
            if isinstance(url, list) and url:
                url = url[0].get("value", "") if isinstance(url[0], dict) else url[0]
            elif isinstance(url, dict):
                url = url.get("value", "")

            # Extract abstract
            abstract = record.get("abstract", None)

            # Create metadata
            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher=publication_name,
                publication_year=year,
                isbn=isbn,
                doi=doi,
                abstract=abstract,
                keywords=record.get("subjects", []),
                page_count=record.get("pageCount"),
                language=record.get("language", "en"),
                license=record.get("copyright"),
                source_publisher="Springer Nature",
                raw_metadata=record,
            )

            # Assess therapeutic relevance
            relevance_score = self.assess_therapeutic_relevance(
                title=title,
                abstract=abstract or "",
                keywords=record.get("subjects", []),
            )

            metadata.therapeutic_relevance_score = relevance_score

            # Assign to stage based on relevance
            if relevance_score >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif relevance_score >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata

        except Exception as e:
            logger.warning(f"Error parsing Springer record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """
        Get detailed metadata for a specific book

        Args:
            book_id: Springer book ID or DOI

        Returns:
            BookMetadata object or None
        """
        if not self._auth_token:
            logger.error("Springer API key not set")
            return None

        try:
            # If it's a DOI, use it directly
            if book_id.startswith("10."):
                url = f"{self.api_base_url}/metadata/json"
                params = {"q": f"doi:{book_id}", "api_key": self._auth_token}
            else:
                url = f"{self.api_base_url}/metadata/json/{book_id}"
                params = {"api_key": self._auth_token}

            response = self.session.get(url, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"Springer metadata error: {response.status_code}")
                return None

            data = response.json()
            records = data.get("records", [])

            if records:
                return self._parse_springer_record(records[0], "")

            return None

        except Exception as e:
            logger.error(f"Error getting Springer metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """
        Get book content (Note: Springer typically requires institutional access)

        Args:
            book_id: Springer book ID or DOI
            format: Desired format (PDF, EPUB, HTML)

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Springer book content typically requires institutional access. "
            "This method returns metadata only."
        )

        # Get metadata instead
        metadata = self.get_book_metadata(book_id)

        if metadata:
            return BookContent(
                book_id=book_id,
                format=format,
                content=None,  # Content not available without institutional access
                metadata=metadata.raw_metadata,
                download_url=metadata.raw_metadata.get("url")
                if metadata.raw_metadata
                else None,
                file_size=None,
                checksum=None,
            )

        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """
        Get chapter content (Note: Requires institutional access)

        Args:
            book_id: Springer book ID
            chapter_id: Chapter ID
            format: Desired format

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Springer chapter content requires institutional access. "
            "Metadata only available through public API."
        )
        return None
