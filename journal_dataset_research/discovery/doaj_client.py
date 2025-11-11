"""
DOAJ (Directory of Open Access Journals) API Client

Implements search integration with DOAJ API for discovering therapeutic
datasets from open access psychology journals.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

import requests

from ai.journal_dataset_research.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class DOAJClient:
    """Client for searching DOAJ for psychology and therapy journals."""

    # Psychology and therapy journal subject categories
    PSYCHOLOGY_SUBJECTS = [
        "Psychology",
        "Psychiatry",
        "Psychotherapy",
        "Counseling",
        "Mental Health",
        "Clinical Psychology",
        "Behavioral Sciences",
    ]

    def __init__(
        self,
        base_url: str = "https://doaj.org/api/v2",
        rate_limit_delay: float = 1.0,  # DOAJ recommends 1 request/second
        page_size: int = 100,
    ):
        """Initialize DOAJ client."""
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.page_size = page_size
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def search_journals(
        self,
        subjects: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        max_journals: int = 50,
    ) -> List[str]:
        """
        Search for psychology/therapy journals in DOAJ.

        Args:
            subjects: List of subject categories to search
            keywords: Optional keywords to filter journals
            max_journals: Maximum number of journals to return

        Returns:
            List of journal ISSNs
        """
        if subjects is None:
            subjects = self.PSYCHOLOGY_SUBJECTS

        self._rate_limit()

        # Build search query
        query_parts = []
        for subject in subjects:
            query_parts.append(f'subject.exact:"{subject}"')

        if keywords:
            for keyword in keywords:
                query_parts.append(f'title:"{keyword}" OR abstract:"{keyword}"')

        query = " AND ".join(query_parts)

        params = {
            "q": query,
            "pageSize": min(self.page_size, max_journals),
            "page": 1,
        }

        try:
            url = f"{self.base_url}/search/journals"
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            journals = []
            if "results" in data:
                for journal in data["results"]:
                    if "bibjson" in journal and "issn" in journal["bibjson"]:
                        issn = journal["bibjson"]["issn"]
                        if isinstance(issn, list) and issn:
                            journals.append(issn[0])
                        elif isinstance(issn, str):
                            journals.append(issn)

            logger.info(f"Found {len(journals)} journals in DOAJ")
            return journals[:max_journals]

        except Exception as e:
            logger.error(f"Error searching DOAJ journals: {e}", exc_info=True)
            return []

    def search_articles(
        self,
        journal_issn: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        therapeutic_keywords: Optional[List[str]] = None,
        max_results: int = 100,
    ) -> List[DatasetSource]:
        """
        Search for articles in DOAJ, optionally filtered by journal.

        Args:
            journal_issn: Optional journal ISSN to filter by
            keywords: General search keywords
            therapeutic_keywords: Keywords related to therapy/datasets
            max_results: Maximum number of results to return

        Returns:
            List of DatasetSource objects
        """
        all_sources = []
        page = 1

        # Build search query
        query_parts = []

        if journal_issn:
            query_parts.append(f'issn:"{journal_issn}"')

        # Add therapeutic keywords
        if therapeutic_keywords:
            for keyword in therapeutic_keywords:
                query_parts.append(f'title:"{keyword}" OR abstract:"{keyword}"')

        # Add dataset-related terms
        dataset_terms = [
            "dataset",
            "data availability",
            "supplementary data",
            "conversation",
            "transcript",
            "therapy session",
        ]
        for term in dataset_terms:
            query_parts.append(f'title:"{term}" OR abstract:"{term}"')

        if keywords:
            for keyword in keywords:
                query_parts.append(f'title:"{keyword}" OR abstract:"{keyword}"')

        query = " AND ".join(query_parts) if query_parts else "*"

        try:
            while len(all_sources) < max_results:
                self._rate_limit()

                params = {
                    "q": query,
                    "pageSize": min(self.page_size, max_results - len(all_sources)),
                    "page": page,
                }

                url = f"{self.base_url}/search/articles"
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if "results" not in data or not data["results"]:
                    break

                for article in data["results"]:
                    try:
                        source = self._parse_article(article)
                        if source:
                            all_sources.append(source)
                            if len(all_sources) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Error parsing DOAJ article: {e}")
                        continue

                # Check if there are more pages
                total = data.get("total", 0)
                if len(all_sources) >= total or len(all_sources) >= max_results:
                    break

                page += 1

            logger.info(f"Found {len(all_sources)} articles from DOAJ")
            return all_sources[:max_results]

        except Exception as e:
            logger.error(f"Error searching DOAJ articles: {e}", exc_info=True)
            return []

    def _parse_article(self, article: dict) -> Optional[DatasetSource]:
        """Parse a DOAJ article into a DatasetSource."""
        try:
            bibjson = article.get("bibjson", {})
            id = article.get("id", "")

            # Extract title
            title = bibjson.get("title", "Untitled")
            if isinstance(title, list):
                title = title[0] if title else "Untitled"

            # Extract authors
            authors = []
            author_list = bibjson.get("author", [])
            for author in author_list:
                if isinstance(author, dict):
                    name = author.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)

            # Extract publication date
            pub_date = self._parse_date(bibjson)

            # Extract DOI
            doi = None
            identifier_list = bibjson.get("identifier", [])
            for identifier in identifier_list:
                if isinstance(identifier, dict) and identifier.get("type") == "doi":
                    doi = identifier.get("id")
                    break

            # Extract abstract
            abstract = ""
            abstract_list = bibjson.get("abstract", [])
            if abstract_list:
                abstract = abstract_list[0] if isinstance(abstract_list[0], str) else ""

            # Extract keywords
            keywords = []
            keyword_list = bibjson.get("keywords", [])
            for keyword in keyword_list:
                if isinstance(keyword, str):
                    keywords.append(keyword)

            # Extract URL
            link_list = bibjson.get("link", [])
            url = "https://doaj.org/"
            for link in link_list:
                if isinstance(link, dict) and link.get("type") == "fulltext":
                    url = link.get("url", url)
                    break

            # Check for data availability
            data_availability = self._detect_data_availability(abstract, title)

            # Generate source ID
            source_id = f"doaj_{id}" if id else f"doaj_{hash(title)}"

            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type="journal",
                url=url,
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                open_access=True,  # DOAJ is open access
                data_availability=data_availability,
                discovery_date=datetime.now(),
                discovery_method="doaj_manual",
            )

        except Exception as e:
            logger.warning(f"Error parsing DOAJ article: {e}")
            return None

    def _parse_date(self, bibjson: dict) -> datetime:
        """Parse publication date from DOAJ article."""
        try:
            # Try year field
            year = bibjson.get("year")
            if year:
                return datetime(int(year), 1, 1)

            # Try start_date
            start_date = bibjson.get("start_date")
            if start_date:
                return datetime.fromisoformat(start_date.replace("Z", "+00:00"))

        except (ValueError, TypeError, AttributeError):
            pass

        return datetime.now()

    def _detect_data_availability(self, abstract: str, title: str) -> str:
        """Detect data availability from abstract and title."""
        text = (abstract + " " + title).lower()

        if any(
            phrase in text
            for phrase in [
                "data available",
                "dataset available",
                "supplementary data",
                "data repository",
            ]
        ):
            return "available"

        if any(
            phrase in text
            for phrase in [
                "data upon request",
                "contact author",
                "data available from",
            ]
        ):
            return "upon_request"

        if any(phrase in text for phrase in ["restricted", "confidential"]):
            return "restricted"

        return "unknown"

