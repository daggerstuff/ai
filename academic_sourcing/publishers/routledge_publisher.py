"""
Routledge Publisher Integration

Routledge (part of Taylor & Francis Group) publishes psychology, mental health,
and psychotherapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class RoutledgePublisher(BasePublisher):
    """Integration with Routledge for book sourcing"""

    def __init__(self):
        """Initialize Routledge integration"""
        super().__init__(
            name="Routledge",
            api_base_url="https://www.routledge.com/api",
            requires_auth=True,
        )

        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy"],
            "dbt": ["dialectical behavior therapy"],
            "trauma": ["trauma", "PTSD"],
            "psychotherapy": ["psychotherapy", "counseling"],
            "clinical": ["clinical psychology"],
        }

        logger.info("Initialized Routledge integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Routledge API"""
        self._auth_token = api_key
        logger.info("✅ Routledge authentication set")
        return True

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in Routledge catalog"""
        if not self._auth_token:
            logger.error("Routledge API key not set")
            return []

        logger.info(f"🔍 Searching Routledge for: '{query}'")

        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        params = {
            "query": combined_query,
            "productType": "book",
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
                logger.error(f"Routledge API error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("products", [])

            books = []
            for result in results:
                try:
                    metadata = self._parse_record(result, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")

            logger.info(f"✅ Found {len(books)} books from Routledge")
            return books

        except Exception as e:
            logger.error(f"Routledge search error: {e}")
            return []

    def _parse_record(
        self, record: Dict[str, Any], query: str
    ) -> Optional[BookMetadata]:
        """Parse Routledge record"""
        try:
            title = record.get("title", "Unknown Title")
            authors = [
                a.get("name", "") for a in record.get("authors", []) if a.get("name")
            ]
            year = record.get("publicationYear", 0)

            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Routledge",
                publication_year=year,
                isbn=record.get("isbn"),
                doi=record.get("doi"),
                abstract=record.get("description"),
                source_publisher="Routledge",
                raw_metadata=record,
            )

            relevance = self.assess_therapeutic_relevance(
                title=title, abstract=record.get("description", ""), keywords=[]
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
            headers = {"Authorization": f"Bearer {self._auth_token}"}
            response = self.session.get(
                f"{self.api_base_url}/products/{book_id}", headers=headers, timeout=10
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
        logger.warning("Routledge content requires institutional access")
        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get chapter content"""
        return None
