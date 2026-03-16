"""
Wiley Publisher Integration

Provides integration with Wiley for sourcing psychology and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class WileyPublisher(BasePublisher):
    """Integration with Wiley for book sourcing"""

    def __init__(self):
        """Initialize Wiley publisher integration"""
        super().__init__(
            name="Wiley",
            api_base_url="https://api.wiley.com/onlinelibrary/tdm/v1",
            requires_auth=True,
        )

        # Wiley-specific therapeutic topics
        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "CBT", "cognitive therapy"],
            "dbt": ["dialectical behavior therapy", "DBT"],
            "trauma": ["trauma", "PTSD", "post-traumatic stress"],
            "anxiety": ["anxiety", "panic", "phobia", "GAD"],
            "depression": ["depression", "MDD", "depressive disorder"],
            "psychotherapy": ["psychotherapy", "counseling", "therapy"],
            "clinical": ["clinical psychology", "psychopathology"],
            "neuroscience": ["neuropsychology", "neuroscience"],
        }

        logger.info("Initialized Wiley publisher integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Wiley API"""
        self._auth_token = api_key

        try:
            test_url = f"{self.api_base_url}/articles"
            headers = {"Wiley-TDM-Client-Token": api_key}
            params = {"query": "psychology", "pageSize": 1}

            response = self.session.get(
                test_url, headers=headers, params=params, timeout=10
            )

            if response.status_code == 200:
                logger.info("✅ Wiley authentication successful")
                return True
            else:
                logger.error(f"Wiley auth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Wiley authentication error: {e}")
            return False

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in Wiley's catalog"""
        if not self._auth_token:
            logger.error("Wiley API key not set")
            return []

        logger.info(f"🔍 Searching Wiley for: '{query}'")

        # Build query
        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])

        headers = {"Wiley-TDM-Client-Token": self._auth_token}
        params = {
            "query": combined_query,
            "publicationType": "book",
            "pageSize": limit,
            "startPage": offset // limit + 1,
        }

        if year_range:
            start_year, end_year = year_range
            params["startYear"] = start_year
            params["endYear"] = end_year

        try:
            url = f"{self.api_base_url}/articles"
            response = self.session.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"Wiley API error: {response.status_code}")
                return []

            data = response.json()
            items = data.get("items", [])

            books = []
            for item in items:
                try:
                    metadata = self._parse_wiley_record(item, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing Wiley record: {e}")
                    continue

            logger.info(f"✅ Found {len(books)} books from Wiley")
            return books

        except Exception as e:
            logger.error(f"Wiley search error: {e}")
            return []

    def _parse_wiley_record(
        self, record: Dict[str, Any], original_query: str
    ) -> Optional[BookMetadata]:
        """Parse a Wiley API record"""
        try:
            title = record.get("title", "Unknown Title")

            # Extract authors
            authors = []
            for author in record.get("authors", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)

            # Extract year
            pub_date = record.get("publicationDate", "")
            year = 0
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                except (ValueError, IndexError):
                    pass

            # Extract identifiers
            doi = record.get("doi")
            isbn = record.get("isbn")

            # Create metadata
            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Wiley",
                publication_year=year,
                isbn=isbn,
                doi=doi,
                abstract=record.get("abstract"),
                keywords=record.get("keywords", []),
                language=record.get("language", "en"),
                source_publisher="Wiley",
                raw_metadata=record,
            )

            # Score relevance
            relevance_score = self.assess_therapeutic_relevance(
                title=title,
                abstract=record.get("abstract", ""),
                keywords=record.get("keywords", []),
            )

            metadata.therapeutic_relevance_score = relevance_score

            if relevance_score >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif relevance_score >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata

        except Exception as e:
            logger.warning(f"Error parsing Wiley record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """Get detailed metadata for a specific book"""
        if not self._auth_token:
            logger.error("Wiley API key not set")
            return None

        try:
            headers = {"Wiley-TDM-Client-Token": self._auth_token}
            url = f"{self.api_base_url}/articles/{book_id}"

            response = self.session.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.error(f"Wiley metadata error: {response.status_code}")
                return None

            data = response.json()
            return self._parse_wiley_record(data, "")

        except Exception as e:
            logger.error(f"Error getting Wiley metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get book content (requires institutional access)"""
        logger.warning("Wiley content requires institutional access")

        metadata = self.get_book_metadata(book_id)
        if metadata:
            return BookContent(
                book_id=book_id,
                format=format,
                content=None,
                metadata=metadata.raw_metadata,
                download_url=None,
                file_size=None,
                checksum=None,
            )
        return None

    def get_chapter_content(
        self, book_id: str, chapter_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get chapter content (requires institutional access)"""
        logger.warning("Wiley chapter content requires institutional access")
        return None
