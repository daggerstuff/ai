"""
Taylor & Francis Publisher Integration

Provides integration with Taylor & Francis for sourcing psychology and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class TaylorFrancisPublisher(BasePublisher):
    """Integration with Taylor & Francis for book sourcing"""

    def __init__(self):
        """Initialize Taylor & Francis publisher integration"""
        super().__init__(
            name="Taylor & Francis",
            api_base_url="https://www.tandfonline.com/api",
            requires_auth=True,
        )

        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "CBT"],
            "dbt": ["dialectical behavior therapy", "DBT"],
            "trauma": ["trauma", "PTSD", "traumatic stress"],
            "anxiety": ["anxiety", "panic", "phobia"],
            "depression": ["depression", "mood disorders"],
            "psychotherapy": ["psychotherapy", "counseling"],
            "clinical": ["clinical psychology"],
        }

        logger.info("Initialized Taylor & Francis integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Taylor & Francis API"""
        self._auth_token = api_key

        try:
            headers = {"Authorization": f"Bearer {api_key}"}
            params = {"query": "psychology", "rows": 1}

            response = self.session.get(
                f"{self.api_base_url}/search",
                headers=headers,
                params=params,
                timeout=10,
            )

            if response.status_code == 200:
                logger.info("✅ Taylor & Francis authentication successful")
                return True
            else:
                logger.error(f"Taylor & Francis auth failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Taylor & Francis auth error: {e}")
            return False

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in Taylor & Francis catalog"""
        if not self._auth_token:
            logger.error("Taylor & Francis API key not set")
            return []

        logger.info(f"🔍 Searching Taylor & Francis for: '{query}'")

        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])
        headers = {"Authorization": f"Bearer {self._auth_token}"}
        params = {
            "query": combined_query,
            "contentType": "book",
            "rows": limit,
            "start": offset,
        }

        if year_range:
            params["publicationDateFrom"] = f"{year_range[0]}-01-01"
            params["publicationDateTo"] = f"{year_range[1]}-12-31"

        try:
            response = self.session.get(
                f"{self.api_base_url}/search",
                headers=headers,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                logger.error(f"Taylor & Francis API error: {response.status_code}")
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

            logger.info(f"✅ Found {len(books)} books from Taylor & Francis")
            return books

        except Exception as e:
            logger.error(f"Taylor & Francis search error: {e}")
            return []

    def _parse_record(
        self, record: Dict[str, Any], query: str
    ) -> Optional[BookMetadata]:
        """Parse Taylor & Francis record"""
        try:
            title = record.get("title", "Unknown Title")
            authors = [
                a.get("name", "") for a in record.get("authors", []) if a.get("name")
            ]

            pub_date = record.get("publicationDate", "")
            year = int(pub_date.split("-")[0]) if pub_date else 0

            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Taylor & Francis",
                publication_year=year,
                isbn=record.get("isbn"),
                doi=record.get("doi"),
                abstract=record.get("abstract"),
                keywords=record.get("keywords", []),
                source_publisher="Taylor & Francis",
                raw_metadata=record,
            )

            relevance = self.assess_therapeutic_relevance(
                title=title,
                abstract=record.get("abstract", ""),
                keywords=record.get("keywords", []),
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
        logger.warning("Taylor & Francis content requires institutional access")
        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get chapter content (requires institutional access)"""
        return None
