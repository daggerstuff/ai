"""
SAGE Publications Publisher Integration

Provides integration with SAGE for sourcing psychology and therapy books.
SAGE is a major publisher of social science and psychology research.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class SAGEPublisher(BasePublisher):
    """Integration with SAGE Publications for book sourcing"""

    def __init__(self):
        """Initialize SAGE publisher integration"""
        super().__init__(
            name="SAGE Publications",
            api_base_url="https://sk.sagepub.com/api",
            requires_auth=True,
        )

        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "cognitive therapy"],
            "dbt": ["dialectical behavior therapy"],
            "trauma": ["trauma", "PTSD", "traumatic stress"],
            "anxiety": ["anxiety", "panic", "social anxiety"],
            "depression": ["depression", "depressive disorders"],
            "psychotherapy": ["psychotherapy", "counseling", "therapy"],
            "clinical": ["clinical psychology", "psychopathology"],
            "social": ["social psychology", "interpersonal"],
        }

        logger.info("Initialized SAGE Publications integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with SAGE API"""
        self._auth_token = api_key

        try:
            headers = {"api-key": api_key}
            params = {"q": "psychology", "limit": 1}

            response = self.session.get(
                f"{self.api_base_url}/search",
                headers=headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("✅ SAGE authentication successful")
                return True
            else:
                logger.error(f"SAGE auth failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"SAGE auth error: {e}")
            return False

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in SAGE catalog"""
        if not self._auth_token:
            logger.error("SAGE API key not set")
            return []

        logger.info(f"🔍 Searching SAGE for: '{query}'")

        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])
        headers = {"api-key": self._auth_token}
        params = {
            "q": combined_query,
            "contentType": "book",
            "limit": limit,
            "offset": offset,
        }

        if year_range:
            params["yearFrom"] = year_range[0]
            params["yearTo"] = year_range[1]

        try:
            response = self.session.get(
                f"{self.api_base_url}/search",
                headers=headers,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                logger.error(f"SAGE API error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])

            books = []
            for result in results:
                try:
                    metadata = self._parse_record(result, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")

            logger.info(f"✅ Found {len(books)} books from SAGE")
            return books

        except Exception as e:
            logger.error(f"SAGE search error: {e}")
            return []

    def _parse_record(
        self, record: Dict[str, Any], query: str
    ) -> Optional[BookMetadata]:
        """Parse SAGE record"""
        try:
            title = record.get("title", "Unknown Title")
            authors = [
                a.get("name", "")
                for a in record.get("contributors", [])
                if a.get("name")
            ]

            year = record.get("publicationYear", 0)
            if isinstance(year, str):
                year = int(year) if year.isdigit() else 0

            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="SAGE Publications",
                publication_year=year,
                isbn=record.get("isbn"),
                doi=record.get("doi"),
                abstract=record.get("description"),
                keywords=record.get("subjects", []),
                source_publisher="SAGE Publications",
                raw_metadata=record,
            )

            relevance = self.assess_therapeutic_relevance(
                title=title,
                abstract=record.get("description", ""),
                keywords=record.get("subjects", []),
            )
            metadata.therapeutic_relevance_score = relevance

            if relevance >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif relevance >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata
        except Exception as e:
            logger.warning(f"Error parsing record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """Get book metadata"""
        if not self._auth_token:
            return None
        try:
            headers = {"api-key": self._auth_token}
            response = self.session.get(
                f"{self.api_base_url}/books/{book_id}", headers=headers, timeout=10
            )
            if response.status_code == 200:
                return self._parse_record(response.json(), "")
            return None
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get book content (requires institutional access)"""
        logger.warning("SAGE content requires institutional access")
        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get chapter content (requires institutional access)"""
        return None
