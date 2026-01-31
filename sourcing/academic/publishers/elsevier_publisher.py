"""
Elsevier Publisher Integration

Provides integration with Elsevier for sourcing psychology and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class ElsevierPublisher(BasePublisher):
    """Integration with Elsevier for book sourcing"""

    def __init__(self):
        """Initialize Elsevier publisher integration"""
        super().__init__(
            name="Elsevier",
            api_base_url="https://api.elsevier.com/content",
            requires_auth=True,
        )

        # Elsevier-specific therapeutic topics
        self.therapeutic_topics_map = {
            "cbt": ["cognitive behavioral therapy", "cognitive therapy"],
            "dbt": ["dialectical behavior therapy"],
            "trauma": ["trauma", "PTSD", "traumatic stress"],
            "anxiety": ["anxiety disorders", "panic", "phobia"],
            "depression": ["depression", "depressive disorders"],
            "psychotherapy": ["psychotherapy", "counseling"],
            "clinical": ["clinical psychology", "psychopathology"],
            "neuroscience": ["neuropsychology", "neuroscience", "brain"],
        }

        logger.info("Initialized Elsevier publisher integration")

    def authenticate(self, api_key: str) -> bool:
        """Authenticate with Elsevier API"""
        self._auth_token = api_key

        try:
            test_url = f"{self.api_base_url}/search/sciencedirect"
            headers = {"X-ELS-APIKey": api_key}
            params = {"query": "psychology", "count": 1}

            response = self.session.get(
                test_url, headers=headers, params=params, timeout=10
            )

            if response.status_code == 200:
                logger.info("✅ Elsevier authentication successful")
                return True
            else:
                logger.error(f"Elsevier auth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Elsevier authentication error: {e}")
            return False

    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[BookMetadata]:
        """Search for books in Elsevier's catalog"""
        if not self._auth_token:
            logger.error("Elsevier API key not set")
            return []

        logger.info(f"🔍 Searching Elsevier for: '{query}'")

        # Build query
        search_terms = [query]
        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])

        headers = {"X-ELS-APIKey": self._auth_token}
        params = {
            "query": combined_query,
            "contentType": "book",
            "count": limit,
            "start": offset,
        }

        if year_range:
            start_year, end_year = year_range
            params["date"] = f"{start_year}-{end_year}"

        try:
            url = f"{self.api_base_url}/search/sciencedirect"
            response = self.session.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"Elsevier API error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("search-results", {}).get("entry", [])

            books = []
            for result in results:
                try:
                    metadata = self._parse_elsevier_record(result, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing Elsevier record: {e}")
                    continue

            logger.info(f"✅ Found {len(books)} books from Elsevier")
            return books

        except Exception as e:
            logger.error(f"Elsevier search error: {e}")
            return []

    def _parse_elsevier_record(
        self, record: Dict[str, Any], original_query: str
    ) -> Optional[BookMetadata]:
        """Parse an Elsevier API record"""
        try:
            title = record.get("dc:title", "Unknown Title")

            # Extract authors
            authors = []
            creator = record.get("dc:creator", "")
            if creator:
                if isinstance(creator, list):
                    authors = creator
                else:
                    authors = [creator]

            # Extract year
            pub_date = record.get("prism:coverDate", "")
            year = 0
            if pub_date:
                try:
                    year = int(pub_date.split("-")[0])
                except (ValueError, IndexError):
                    pass

            # Extract identifiers
            doi = record.get("prism:doi")
            isbn = record.get("prism:isbn")

            # Create metadata
            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Elsevier",
                publication_year=year,
                isbn=isbn,
                doi=doi,
                abstract=record.get("dc:description"),
                language=record.get("dc:language", "en"),
                source_publisher="Elsevier",
                raw_metadata=record,
            )

            # Score relevance
            relevance_score = self.assess_therapeutic_relevance(
                title=title, abstract=record.get("dc:description", ""), keywords=[]
            )

            metadata.therapeutic_relevance_score = relevance_score

            if relevance_score >= 0.7:
                metadata.stage_assignment = "stage2_therapeutic_expertise"
            elif relevance_score >= 0.4:
                metadata.stage_assignment = "stage1_foundation"

            return metadata

        except Exception as e:
            logger.warning(f"Error parsing Elsevier record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """Get detailed metadata for a specific book"""
        if not self._auth_token:
            logger.error("Elsevier API key not set")
            return None

        try:
            headers = {"X-ELS-APIKey": self._auth_token}

            # Try by DOI or PII
            if book_id.startswith("10."):
                url = f"{self.api_base_url}/article/doi/{book_id}"
            else:
                url = f"{self.api_base_url}/article/pii/{book_id}"

            response = self.session.get(url, headers=headers, timeout=10)

            if response.status_code != 200:
                logger.error(f"Elsevier metadata error: {response.status_code}")
                return None

            data = response.json()
            core_data = data.get("full-text-retrieval-response", {}).get("coredata", {})

            return self._parse_elsevier_record(core_data, "")

        except Exception as e:
            logger.error(f"Error getting Elsevier metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """Get book content (requires institutional access)"""
        logger.warning("Elsevier content requires institutional access")

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
        logger.warning("Elsevier chapter content requires institutional access")
        return None
