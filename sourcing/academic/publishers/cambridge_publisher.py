"""
Cambridge University Press Publisher Integration

Provides integration with Cambridge University Press for sourcing psychology
and therapy books.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .base_publisher import BasePublisher, BookContent, BookFormat, BookMetadata

logger = logging.getLogger(__name__)


class CambridgePublisher(BasePublisher):
    """Integration with Cambridge University Press for book sourcing"""

    def __init__(self):
        """Initialize Cambridge University Press integration"""
        super().__init__(
            name="Cambridge University Press",
            api_base_url="https://www.cambridge.org/core/api",
            requires_auth=True,
        )

        # Cambridge-specific therapeutic topics mapping
        self.therapeutic_topics_map = {
            "cbt": [
                "cognitive behavioral therapy",
                "cognitive therapy",
                "behavioral interventions",
            ],
            "dbt": ["dialectical behavior therapy", "borderline personality"],
            "trauma": [
                "trauma",
                "ptsd",
                "post-traumatic stress disorder",
                "complex trauma",
            ],
            "anxiety": ["anxiety", "panic", "phobia", "worry", "fear"],
            "depression": [
                "depression",
                "depressive disorders",
                "mood disorders",
                "bipolar",
            ],
            "psychotherapy": ["psychotherapy", "counseling", "therapy", "therapeutic"],
            "clinical": ["clinical psychology", "psychopathology", "mental disorders"],
            "neuroscience": ["neuropsychology", "cognitive neuroscience", "brain"],
            "developmental": [
                "child psychology",
                "adolescent",
                "developmental psychology",
            ],
            "social": ["social psychology", "interpersonal", "relationships"],
        }

        logger.info("Initialized Cambridge University Press integration")

    def authenticate(self, api_key: str) -> bool:
        """
        Authenticate with Cambridge API

        Args:
            api_key: Cambridge API key

        Returns:
            True if authentication successful
        """
        self._auth_token = api_key

        # Test authentication
        try:
            test_url = f"{self.api_base_url}/search"
            headers = {"api-key": api_key}
            params = {"query": "psychology", "limit": 1}

            response = self.session.get(
                test_url, headers=headers, params=params, timeout=10
            )

            if response.status_code == 200:
                logger.info("✅ Cambridge authentication successful")
                return True
            else:
                logger.error(f"Cambridge auth failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Cambridge authentication error: {e}")
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
        Search for books in Cambridge's catalog

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
            logger.error("Cambridge API key not set. Call authenticate() first.")
            return []

        logger.info(f"🔍 Searching Cambridge for: '{query}'")

        # Build enhanced query
        search_terms = [query]

        if therapeutic_topics:
            for topic in therapeutic_topics:
                if topic in self.therapeutic_topics_map:
                    search_terms.extend(self.therapeutic_topics_map[topic])

        combined_query = " OR ".join(search_terms[:5])

        # API parameters
        headers = {"api-key": self._auth_token}
        params = {
            "query": combined_query,
            "limit": limit,
            "offset": offset,
            "productTypes": "BOOK",
        }

        # Add year filter if specified
        if year_range:
            start_year, end_year = year_range
            params["dateFrom"] = start_year
            params["dateTo"] = end_year

        try:
            url = f"{self.api_base_url}/search"
            response = self.session.get(url, headers=headers, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"Cambridge API error: {response.status_code}")
                return []

            data = response.json()
            items = data.get("items", [])

            books = []
            for item in items:
                try:
                    metadata = self._parse_cambridge_record(item, query)
                    if metadata:
                        books.append(metadata)
                except Exception as e:
                    logger.warning(f"Error parsing Cambridge record: {e}")
                    continue

            logger.info(f"✅ Found {len(books)} books from Cambridge")
            return books

        except Exception as e:
            logger.error(f"Cambridge search error: {e}")
            return []

    def _parse_cambridge_record(
        self, record: Dict[str, Any], original_query: str
    ) -> Optional[BookMetadata]:
        """Parse a Cambridge API record into BookMetadata"""
        try:
            # Extract basic info
            title = record.get("title", "Unknown Title")

            # Extract authors
            authors = []
            creators = record.get("creators", [])
            for creator in creators:
                if isinstance(creator, dict):
                    name = creator.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(creator, str):
                    authors.append(creator)

            # Extract publication info
            pub_date = record.get("publishedDate", "")
            pub_year = 0
            if pub_date:
                try:
                    pub_year = int(pub_date.split("-")[0])
                except (ValueError, IndexError):
                    pass

            # Extract identifiers
            isbn = None
            doi = None

            identifiers = record.get("identifiers", [])
            for identifier in identifiers:
                if isinstance(identifier, dict):
                    id_type = identifier.get("type", "").lower()
                    id_value = identifier.get("value", "")

                    if "isbn" in id_type:
                        isbn = id_value
                    elif "doi" in id_type:
                        doi = id_value

            # Extract abstract
            abstract = record.get("abstract") or record.get("description")

            # Extract subjects/keywords
            subjects = record.get("subjects", [])
            keywords = []
            for subject in subjects:
                if isinstance(subject, dict):
                    keywords.append(subject.get("name", ""))
                elif isinstance(subject, str):
                    keywords.append(subject)

            # Create metadata
            metadata = BookMetadata(
                title=title,
                authors=authors,
                publisher="Cambridge University Press",
                publication_year=pub_year,
                isbn=isbn,
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                page_count=record.get("numberOfPages"),
                language=record.get("language", "en"),
                license=record.get("license"),
                source_publisher="Cambridge University Press",
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
            logger.warning(f"Error parsing Cambridge record: {e}")
            return None

    def get_book_metadata(self, book_id: str) -> Optional[BookMetadata]:
        """
        Get detailed metadata for a specific book

        Args:
            book_id: Cambridge book ID or DOI

        Returns:
            BookMetadata object or None
        """
        if not self._auth_token:
            logger.error("Cambridge API key not set")
            return None

        try:
            headers = {"api-key": self._auth_token}

            # If it's a DOI, search by DOI
            if book_id.startswith("10."):
                url = f"{self.api_base_url}/search"
                params = {"query": f"doi:{book_id}"}
            else:
                url = f"{self.api_base_url}/products/{book_id}"
                params = {}

            response = self.session.get(url, headers=headers, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"Cambridge metadata error: {response.status_code}")
                return None

            data = response.json()

            # Handle search response vs direct product response
            if "items" in data and data["items"]:
                return self._parse_cambridge_record(data["items"][0], "")
            else:
                return self._parse_cambridge_record(data, "")

        except Exception as e:
            logger.error(f"Error getting Cambridge metadata: {e}")
            return None

    def get_book_content(
        self, book_id: str, format: BookFormat = BookFormat.PDF
    ) -> Optional[BookContent]:
        """
        Get book content (Note: Requires institutional access)

        Args:
            book_id: Cambridge book ID
            format: Desired format (PDF, EPUB, HTML)

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Cambridge book content requires institutional access. "
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
            book_id: Cambridge book ID
            chapter_id: Chapter ID
            format: Desired format

        Returns:
            BookContent object or None
        """
        logger.warning(
            "Cambridge chapter content requires institutional access. "
            "Metadata only available through public API."
        )
        return None
