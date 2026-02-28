"""
Guilford Press Publisher Integration

Guilford Press specializes in clinical psychology, psychiatry, and mental health.
Major publisher of CBT, DBT, and evidence-based therapy resources.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class GuilfordPublisher(BasePublisher):
    """Integration with Guilford Press for book sourcing"""

    def __init__(self):
        """Initialize Guilford Press integration"""
        super().__init__(
            name="Guilford Press",
            api_base_url="https://www.guilford.com/api",
            requires_auth=False,  # Guilford may have public search
        )

        # Guilford specializes in clinical/therapeutic content
        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "CBT", "Beck", "cognitive therapy"],
            "dbt": [
                "dialectical behavior therapy",
                "DBT",
                "Linehan",
                "emotion regulation",
            ],
            "trauma": ["trauma", "PTSD", "complex trauma", "trauma-focused"],
            "anxiety": ["anxiety", "panic", "OCD", "phobia"],
            "depression": ["depression", "mood disorders", "bipolar"],
            "personality": ["personality disorders", "borderline", "narcissistic"],
            "child": ["child therapy", "adolescent", "pediatric"],
            "couples": ["couples therapy", "marriage", "relationship"],
        }

        logger.info("Initialized Guilford Press integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate (may not be required for Guilford)"""
        self._auth_token = api_key
        logger.info("✅ Guilford authentication set")
        return True

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in Guilford catalog"""
        logger.info(f"🔍 Searching Guilford for: '{query}'")

        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " ".join(search_terms[:5])

        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        params = {
            "q": combined_query,
            "type": "book",
            "limit": limit,
            "offset": offset,
        }

        if year_range:
            params["year_from"] = year_range[0]
            params["year_to"] = year_range[1]

        try:
            response = self.session.get(
                f"{self.api_base_url}/search",
                headers=headers,
                params=params,
                timeout=15,
            )

            if response.status_code != 200:
                logger.error(f"Guilford API error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("books", [])

            books = []
            for result in results:
                try:
                    metadata = self._parse_record(result, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing record: {e}")

            logger.info(f"✅ Found {len(books)} books from Guilford")
            return books

        except Exception as e:
            logger.error(f"Guilford search error: {e}")
            return []

    def _parse_record(
        self, record: Dict[str, Any], query: str
    ) -> Optional[BookMetadata]:
        """Parse Guilford record"""
        try:
            title = record.get("title", "Unknown Title")
            authors = record.get("authors", [])
            if isinstance(authors, str):
                authors = [authors]

            year = record.get("publication_year", 0)

            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Guilford Press",
                publication_year=year,
                isbn=record.get("isbn"),
                abstract=record.get("description"),
                keywords=record.get("topics", []),
                source_publisher="Guilford Press",
                raw_metadata=record,
            )

            # Guilford books are typically highly relevant to therapy
            relevance = self.assess_therapeutic_relevance(
                title=title,
                abstract=record.get("description", ""),
                keywords=record.get("topics", []),
            )
            # Boost relevance for Guilford (specialized publisher)
            metadata.therapeutic_relevance_score = min(relevance + 0.2, 1.0)

            if metadata.therapeutic_relevance_score >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif metadata.therapeutic_relevance_score >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata
        except Exception as e:
            logger.warning(f"Error parsing record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """Get book metadata"""
        try:
            headers = {}
            if self._auth_token:
                headers["Authorization"] = f"Bearer {self._auth_token}"

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
        """Get book content (typically requires purchase)"""
        logger.warning("Guilford content typically requires purchase")
        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get chapter content"""
        return None
