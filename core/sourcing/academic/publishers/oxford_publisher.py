"""
Oxford University Press Publisher Integration

Provides integration with Oxford University Press for sourcing psychology
and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class OxfordPublisher(BasePublisher):
    """Integration with Oxford University Press for book sourcing"""

    def __init__(self):
        """Initialize Oxford University Press integration"""
        super().__init__(
            name="Oxford University Press",
            api_base_url="https://academic.oup.com/api",
            requires_auth=True,
        )

        # Oxford-specific therapeutic topics mapping
        self.therapeutic_topics_map = {
            "cbt": [
                "cognitive behavioral therapy",
                "cognitive therapy",
                "behavioral activation",
            ],
            "dbt": ["dialectical behavior therapy", "emotion regulation"],
            "trauma": [
                "trauma therapy",
                "ptsd",
                "post-traumatic stress",
                "trauma-focused",
            ],
            "anxiety": [
                "anxiety disorders",
                "panic disorder",
                "social anxiety",
                "generalized anxiety",
            ],
            "depression": ["depression", "major depressive disorder", "mood disorders"],
            "personality": [
                "personality disorders",
                "borderline personality",
                "narcissistic",
            ],
            "psychotherapy": [
                "psychotherapy",
                "counseling",
                "therapeutic relationship",
            ],
            "clinical": ["clinical psychology", "clinical assessment", "diagnosis"],
            "neuroscience": ["neuropsychology", "neuroscience", "brain disorders"],
            "child": ["child psychology", "adolescent psychology", "developmental"],
        }

        logger.info("Initialized Oxford University Press integration")

    def authenticate(self, api_key: str) -> bool:
        """
        Authenticate with Oxford API

        Args:
            api_key: Oxford API key

        Returns:
            True if authentication successful
        """
        self._auth_token = api_key

        # Test authentication with a simple query
        try:
            test_url = f"{self.api_base_url}/search"
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"q": "psychology", "limit": 1}

            response = self.session.get(
                test_url, headers=headers, params=params, timeout=10
            )

            if response.status_code == 200:
                logger.info("✅ Oxford authentication successful")
                return True
            else:
                logger.error(f"Oxford auth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Oxford authentication error: {e}")
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
        Search for books in Oxford's catalog

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
            logger.error("Oxford API key not set. Call authenticate() first.")
            return []

        logger.info(f"🔍 Searching Oxford for: '{query}'")

        # Build enhanced query
        search_terms = [query]

        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(f'"{term}"' for term in search_terms[:5])

        # API parameters
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        params = {
            "q": combined_query,
            "limit": limit,
            "offset": offset,
            "contentType": "book",
        }

        # Add year filter if specified
        if year_range:
            start_year, end_year = year_range
            params["fromYear"] = start_year
            params["toYear"] = end_year

        try:
            url = f"{self.api_base_url}/search"
            response = self.session.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"Oxford API error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])

            books = []
            for result in results:
                try:
                    metadata = self._parse_oxford_record(result, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing Oxford record: {e}")
                    continue

            logger.info(f"✅ Found {len(books)} books from Oxford")
            return books

        except Exception as e:
            logger.error(f"Oxford search error: {e}")
            return []

    def _parse_oxford_record(
        self, record: Dict[str, Any], original_query: str
    ) -> Optional[BookMetadata]:
        """Parse an Oxford API record into BookMetadata"""
        try:
            # Extract basic info
            title = record.get("title", "Unknown Title")

            # Extract authors
            authors = []
            contributors = record.get("contributors", [])
            for contributor in contributors:
                if isinstance(contributor, dict):
                    name = contributor.get("name", "")
                    role = contributor.get("role", "").lower()
                    # Include authors and editors
                    if role in ["author", "editor", "contributor"] and name:
                        authors.append(name)
                elif isinstance(contributor, str):
                    authors.append(contributor)

            # Extract publication info
            pub_year = record.get("publicationYear", 0)
            if isinstance(pub_year, str):
                try:
                    pub_year = int(pub_year)
                except ValueError:
                    pub_year = 0

            # Extract identifiers
            isbn = record.get("isbn")
            doi = record.get("doi")

            # Extract abstract/description
            abstract = record.get("abstract") or record.get("description")

            # Extract keywords/subjects
            keywords = record.get("subjects", [])
            if isinstance(keywords, str):
                keywords = [keywords]

            # Create metadata
            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Oxford University Press",
                publication_year=pub_year,
                isbn=isbn,
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                page_count=record.get("pageCount"),
                language=record.get("language", "en"),
                license=record.get("license"),
                source_publisher="Oxford University Press",
                raw_metadata=record,
            )

            # Assess therapeutic relevance
            relevance_score = self.assess_therapeutic_relevance(
                title=title, abstract=abstract or "", keywords=keywords
            )

            metadata.therapeutic_relevance_score = relevance_score

            # Assign to stage based on relevance
            if relevance_score >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif relevance_score >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata

        except Exception as e:
            logger.warning(f"Error parsing Oxford record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """
        Get detailed metadata for a specific book

        Args:
            book_id: Oxford book ID or ISBN

        Returns:
            BookMetadata object or None
        """
        if not self._auth_token:
            logger.error("Oxford API key not set")
            return None

        try:
            headers = {"Authorization": f"Bearer {self._auth_token}"}

            # Try by ISBN first, then by ID
            if book_id.replace("-", "").isdigit():
                url = f"{self.api_base_url}/books/isbn/{book_id}"
            else:
                url = f"{self.api_base_url}/books/{book_id}"

            response = self.session.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.error(f"Oxford metadata error: {response.status_code}")
                return None

            data = response.json()
            return self._parse_oxford_record(data, "")

        except Exception as e:
            logger.error(f"Error getting Oxford metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """
        Get book content (Note: Requires institutional access)

        Args:
            book_id: Oxford book ID
            format: Desired format (PDF, EPUB, HTML)

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Oxford book content requires institutional access. "
            "This method returns metadata only."
        )

        # Get metadata instead
        metadata = self.get_book_metadata(book_id)

        if metadata:
            return BookContent(
                book_id=book_id,
                format=format,
                content=None,
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
            book_id: Oxford book ID
            chapter_id: Chapter ID
            format: Desired format

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Oxford chapter content requires institutional access. "
            "Metadata only available through public API."
        )
        return None
