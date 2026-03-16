"""
DOI Resolver and Searcher

Provides DOI resolution and search capabilities using CrossRef, DataCite,
and other DOI registration agencies.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class DOIMetadata:
    """Metadata extracted from DOI resolution"""

    doi: str
    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    publication_type: str  # journal-article, book, book-chapter, etc.
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    abstract: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None
    license: Optional[str] = None
    references_count: int = 0
    citations_count: int = 0
    raw_metadata: Optional[Dict[str, Any]] = None


class DOIResolver:
    """
    Resolve DOIs to metadata using CrossRef and DataCite APIs
    """

    def __init__(self):
        self.crossref_api = "https://api.crossref.org/works"
        self.datacite_api = "https://api.datacite.org/dois"
        self.doi_org = "https://doi.org"

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PixelatedEmpathy/1.0 (Academic Research)"}
        )

        logger.info("Initialized DOI Resolver")

    def resolve(self, doi: str) -> Optional[DOIMetadata]:
        """
        Resolve a DOI to its metadata

        Args:
            doi: DOI string (with or without https://doi.org/ prefix)

        Returns:
            DOIMetadata object or None if resolution fails
        """
        # Clean DOI
        doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")

        logger.info(f"Resolving DOI: {doi}")

        # Try CrossRef first (most common)
        metadata = self._resolve_crossref(doi)
        if metadata:
            return metadata

        # Try DataCite as fallback
        metadata = self._resolve_datacite(doi)
        if metadata:
            return metadata

        logger.warning(f"Could not resolve DOI: {doi}")
        return None

    def _resolve_crossref(self, doi: str) -> Optional[DOIMetadata]:
        """Resolve DOI using CrossRef API"""
        try:
            url = f"{self.crossref_api}/{doi}"
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()
            message = data.get("message", {})

            # Extract authors
            authors = []
            for author in message.get("author", []):
                given = author.get("given", "")
                family = author.get("family", "")
                name = f"{given} {family}".strip()
                if name:
                    authors.append(name)

            # Extract publication year
            year = 0
            pub_date = message.get("published-print") or message.get("published-online")
            if pub_date:
                date_parts = pub_date.get("date-parts", [[]])[0]
                if date_parts:
                    year = date_parts[0]

            # Extract title
            titles = message.get("title", [])
            title = titles[0] if titles else "Unknown Title"

            # Extract journal info
            journal = None
            container_titles = message.get("container-title", [])
            if container_titles:
                journal = container_titles[0]

            # Create metadata
            metadata = DOIMetadata(
                doi=doi,
                title=title,
                authors=authors,
                publisher=message.get("publisher", "Unknown"),
                publication_year=year,
                publication_type=message.get("type", "unknown"),
                journal=journal,
                volume=message.get("volume"),
                issue=message.get("issue"),
                pages=message.get("page"),
                abstract=message.get("abstract"),
                url=f"https://doi.org/{doi}",
                license=message.get("license", [{}])[0].get("URL")
                if message.get("license")
                else None,
                references_count=message.get("references-count", 0),
                citations_count=message.get("is-referenced-by-count", 0),
                raw_metadata=message,
            )

            logger.info(f"✅ Resolved DOI via CrossRef: {title}")
            return metadata

        except Exception as e:
            logger.debug(f"CrossRef resolution failed: {e}")
            return None

    def _resolve_datacite(self, doi: str) -> Optional[DOIMetadata]:
        """Resolve DOI using DataCite API"""
        try:
            url = f"{self.datacite_api}/{doi}"
            response = self.session.get(url, timeout=10)

            if response.status_code != 200:
                return None

            data = response.json()
            attributes = data.get("data", {}).get("attributes", {})

            # Extract authors
            authors = []
            creators = attributes.get("creators", [])
            for creator in creators:
                name = creator.get("name", "")
                if name:
                    authors.append(name)

            # Extract publication year
            year = attributes.get("publicationYear", 0)

            # Create metadata
            metadata = DOIMetadata(
                doi=doi,
                title=attributes.get("titles", [{}])[0].get("title", "Unknown Title"),
                authors=authors,
                publisher=attributes.get("publisher", "Unknown"),
                publication_year=year,
                publication_type=attributes.get("types", {}).get(
                    "resourceTypeGeneral", "unknown"
                ),
                abstract=attributes.get("descriptions", [{}])[0].get("description")
                if attributes.get("descriptions")
                else None,
                url=attributes.get("url", f"https://doi.org/{doi}"),
                raw_metadata=attributes,
            )

            logger.info(f"✅ Resolved DOI via DataCite: {metadata.title}")
            return metadata

        except Exception as e:
            logger.debug(f"DataCite resolution failed: {e}")
            return None

    def batch_resolve(self, dois: List[str]) -> List[DOIMetadata]:
        """
        Resolve multiple DOIs

        Args:
            dois: List of DOI strings

        Returns:
            List of successfully resolved DOIMetadata objects
        """
        logger.info(f"Batch resolving {len(dois)} DOIs")

        results = []
        for doi in dois:
            metadata = self.resolve(doi)
            if metadata:
                results.append(metadata)

        logger.info(f"✅ Successfully resolved {len(results)}/{len(dois)} DOIs")
        return results


class DOISearcher:
    """
    Search for DOIs using CrossRef search API
    """

    def __init__(self):
        self.crossref_api = "https://api.crossref.org/works"

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "PixelatedEmpathy/1.0 (Academic Research)"}
        )

        logger.info("Initialized DOI Searcher")

    def search(
        self, query: str, filters: Optional[Dict[str, Any]] = None, limit: int = 20
    ) -> List[DOIMetadata]:
        """
        Search for DOIs by query

        Args:
            query: Search query
            filters: Optional filters (e.g., {"type": "book", "from-pub-date": "2020"})
            limit: Maximum results

        Returns:
            List of DOIMetadata objects
        """
        logger.info(f"🔍 Searching DOIs for: '{query}'")

        params = {
            "query": query,
            "rows": limit,
        }

        # Add filters
        if filters:
            filter_strings = []
            for key, value in filters.items():
                filter_strings.append(f"{key}:{value}")
            if filter_strings:
                params["filter"] = ",".join(filter_strings)

        try:
            response = self.session.get(self.crossref_api, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"CrossRef search error: {response.status_code}")
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])

            # Use DOIResolver to parse results
            resolver = DOIResolver()
            results = []

            for item in items:
                doi = item.get("DOI")
                if doi:
                    # Parse the item directly (already have the data)
                    metadata = resolver._parse_crossref_item(item)
                    if metadata:
                        results.append(metadata)

            logger.info(f"✅ Found {len(results)} DOIs")
            return results

        except Exception as e:
            logger.error(f"DOI search error: {e}")
            return []

    def search_by_title(self, title: str, limit: int = 10) -> List[DOIMetadata]:
        """Search for DOIs by title"""
        return self.search(
            query=title,
            filters={"type": "book,book-chapter,journal-article"},
            limit=limit,
        )

    def search_by_author(self, author: str, limit: int = 20) -> List[DOIMetadata]:
        """Search for DOIs by author name"""
        return self.search(query=f"author:{author}", limit=limit)

    def search_psychology_books(
        self, query: str, year_from: Optional[int] = None, limit: int = 20
    ) -> List[DOIMetadata]:
        """
        Search for psychology books

        Args:
            query: Search query
            year_from: Optional minimum publication year
            limit: Maximum results

        Returns:
            List of DOIMetadata objects
        """
        filters = {"type": "book,book-chapter"}

        if year_from:
            filters["from-pub-date"] = str(year_from)

        return self.search(
            query=f"{query} psychology therapy mental health",
            filters=filters,
            limit=limit,
        )


# Helper method for DOIResolver
def _parse_crossref_item(self, item: Dict[str, Any]) -> Optional[DOIMetadata]:
    """Parse a CrossRef search result item"""
    try:
        # Extract authors
        authors = []
        for author in item.get("author", []):
            given = author.get("given", "")
            family = author.get("family", "")
            name = f"{given} {family}".strip()
            if name:
                authors.append(name)

        # Extract publication year
        year = 0
        pub_date = item.get("published-print") or item.get("published-online")
        if pub_date:
            date_parts = pub_date.get("date-parts", [[]])[0]
            if date_parts:
                year = date_parts[0]

        # Extract title
        titles = item.get("title", [])
        title = titles[0] if titles else "Unknown Title"

        # Extract journal info
        journal = None
        container_titles = item.get("container-title", [])
        if container_titles:
            journal = container_titles[0]

        doi = item.get("DOI", "")

        # Create metadata
        metadata = DOIMetadata(
            doi=doi,
            title=title,
            authors=authors,
            publisher=item.get("publisher", "Unknown"),
            publication_year=year,
            publication_type=item.get("type", "unknown"),
            journal=journal,
            volume=item.get("volume"),
            issue=item.get("issue"),
            pages=item.get("page"),
            url=f"https://doi.org/{doi}",
            references_count=item.get("references-count", 0),
            citations_count=item.get("is-referenced-by-count", 0),
            raw_metadata=item,
        )

        return metadata

    except Exception as e:
        logger.warning(f"Error parsing CrossRef item: {e}")
        return None


# Add method to DOIResolver class
DOIResolver._parse_crossref_item = _parse_crossref_item
