"""
PubMed Central E-utilities API Client

Implements search integration with NCBI E-utilities API for discovering
therapeutic datasets from PubMed Central.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlencode

import requests
from xml.etree import ElementTree as ET

from ai.sourcing.journal.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class PubMedClient:
    """Client for searching PubMed Central using NCBI E-utilities API."""

    # MeSH terms for mental health and therapy
    MENTAL_HEALTH_MESH_TERMS = [
        "Mental Health",
        "Psychotherapy",
        "Counseling",
        "Cognitive Behavioral Therapy",
        "Depression",
        "Anxiety Disorders",
        "Post-Traumatic Stress Disorder",
        "Substance-Related Disorders",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        search_limit: int = 100,
        rate_limit_delay: float = 0.34,  # NCBI recommends 3 requests/second max
    ):
        """Initialize PubMed client."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.search_limit = search_limit
        self.rate_limit_delay = rate_limit_delay
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting for API requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def _build_search_query(
        self,
        keywords: List[str],
        mesh_terms: Optional[List[str]] = None,
        open_access_only: bool = True,
        has_data: bool = True,
    ) -> str:
        """
        Build PubMed search query with MeSH terms and filters.

        Args:
            keywords: List of search keywords
            mesh_terms: Optional list of MeSH terms (defaults to mental health terms)
            open_access_only: Filter for open access articles
            has_data: Filter for articles with available data

        Returns:
            PubMed search query string
        """
        query_parts = []

        # Add MeSH terms
        if mesh_terms is None:
            mesh_terms = self.MENTAL_HEALTH_MESH_TERMS

        mesh_query = " OR ".join([f'"{term}"[MeSH Terms]' for term in mesh_terms])
        query_parts.append(f"({mesh_query})")

        # Add keywords
        if keywords:
            keyword_query = " OR ".join([f'"{kw}"[Title/Abstract]' for kw in keywords])
            query_parts.append(f"({keyword_query})")

        # Add dataset-related terms
        dataset_terms = [
            "dataset",
            "data availability",
            "supplementary data",
            "conversation transcript",
            "therapy transcript",
        ]
        dataset_query = " OR ".join([f'"{term}"[Title/Abstract]' for term in dataset_terms])
        query_parts.append(f"({dataset_query})")

        # Combine with AND
        base_query = " AND ".join(query_parts)

        # Add filters
        filters = []
        if open_access_only:
            filters.append("open access[filter]")
        if has_data:
            filters.append("has data[filter]")

        if filters:
            filter_str = " AND ".join(filters)
            return f"{base_query} AND {filter_str}"

        return base_query

    def search(
        self,
        keywords: List[str],
        mesh_terms: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        open_access_only: bool = True,
        has_data: bool = True,
    ) -> List[DatasetSource]:
        """
        Search PubMed Central for therapeutic datasets.

        Args:
            keywords: List of search keywords
            mesh_terms: Optional list of MeSH terms
            max_results: Maximum number of results to return
            open_access_only: Filter for open access articles
            has_data: Filter for articles with available data

        Returns:
            List of DatasetSource objects
        """
        max_results = max_results or self.search_limit
        all_sources = []

        try:
            # Step 1: ESearch - Get article IDs
            search_query = self._build_search_query(
                keywords, mesh_terms, open_access_only, has_data
            )
            article_ids = self._esearch(search_query, max_results)

            if not article_ids:
                logger.info("No articles found in PubMed search")
                return []

            # Step 2: EFetch - Get article details in batches
            batch_size = 100  # NCBI allows up to 100 IDs per request
            for i in range(0, len(article_ids), batch_size):
                batch_ids = article_ids[i : i + batch_size]
                sources = self._efetch(batch_ids)
                all_sources.extend(sources)

                if len(all_sources) >= max_results:
                    break

            logger.info(f"Found {len(all_sources)} dataset sources from PubMed")
            return all_sources[:max_results]

        except Exception as e:
            logger.error(f"Error searching PubMed: {e}", exc_info=True)
            return []

    def _esearch(self, query: str, max_results: int) -> List[str]:
        """Execute ESearch to get article IDs."""
        self._rate_limit()

        params = {
            "db": "pmc",  # PubMed Central
            "term": query,
            "retmax": min(max_results, 10000),  # NCBI limit
            "retmode": "json",
            "usehistory": "n",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/esearch.fcgi"
        response = requests.get(url, params=params, timeout=30)

        response.raise_for_status()
        data = response.json()

        if "esearchresult" in data and "idlist" in data["esearchresult"]:
            return data["esearchresult"]["idlist"]

        return []

    def _efetch(self, article_ids: List[str]) -> List[DatasetSource]:
        """Execute EFetch to get article details."""
        self._rate_limit()

        params = {
            "db": "pmc",
            "id": ",".join(article_ids),
            "retmode": "xml",
            "rettype": "abstract",
        }

        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/efetch.fcgi"
        response = requests.get(url, params=params, timeout=30)

        response.raise_for_status()
        return self._parse_pubmed_xml(response.text)

    def _parse_pubmed_xml(self, xml_content: str) -> List[DatasetSource]:
        """Parse PubMed XML response and extract DatasetSource objects."""
        sources = []

        try:
            root = ET.fromstring(xml_content)

            # Handle both <PubmedArticleSet> and direct <PubmedArticle> elements
            articles = root.findall(".//PubmedArticle")
            if not articles:
                articles = root.findall(".//article")

            for article in articles:
                try:
                    source = self._parse_article(article)
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.warning(f"Error parsing article: {e}")
                    continue

        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing PubMed XML: {e}", exc_info=True)

        return sources

    def _parse_article(self, article: ET.Element) -> Optional[DatasetSource]:
        """Parse a single article element into a DatasetSource."""
        try:
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            if title_elem is None:
                title_elem = article.find(".//title-group/article-title")
            title = title_elem.text if title_elem is not None else "Untitled"

            # Extract authors
            authors = []
            author_list = article.findall(".//Author")
            for author in author_list:
                last_name = author.findtext("LastName", "")
                first_name = author.findtext("ForeName", "")
                if last_name or first_name:
                    authors.append(f"{first_name} {last_name}".strip())

            # Extract publication date
            pub_date_elem = article.find(".//PubDate")
            if pub_date_elem is None:
                pub_date_elem = article.find(".//article-meta/pub-date")
            pub_date = self._parse_date(pub_date_elem)

            # Extract DOI
            doi_elem = article.find(".//ArticleId[@IdType='doi']")
            if doi_elem is None:
                doi_elem = article.find(".//article-id[@pub-id-type='doi']")
            doi = doi_elem.text if doi_elem is not None else None

            # Extract abstract
            abstract_elem = article.find(".//Abstract/AbstractText")
            if abstract_elem is None:
                abstract_elem = article.find(".//abstract/p")
            abstract = abstract_elem.text if abstract_elem is not None else ""

            # Extract keywords
            keywords = []
            keyword_list = article.findall(".//Keyword")
            for keyword in keyword_list:
                if keyword.text:
                    keywords.append(keyword.text)

            # Extract URL (PMC link)
            pmc_id_elem = article.find(".//ArticleId[@IdType='pmc']")
            if pmc_id_elem is not None:
                pmc_id = pmc_id_elem.text
                url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id}/"
            else:
                # Fallback to PubMed link
                pmid_elem = article.find(".//ArticleId[@IdType='pubmed']")
                if pmid_elem is not None:
                    pmid = pmid_elem.text
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                else:
                    url = "https://www.ncbi.nlm.nih.gov/pmc/"

            # Check for data availability indicators
            data_availability = self._detect_data_availability(abstract, title)

            # Generate source ID
            source_id = f"pubmed_{pmc_id_elem.text if pmc_id_elem is not None else pmid_elem.text if pmid_elem is not None else 'unknown'}"

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
                open_access=True,  # PMC is open access
                data_availability=data_availability,
                discovery_date=datetime.now(),
                discovery_method="pubmed_search",
            )

        except Exception as e:
            logger.warning(f"Error parsing article element: {e}")
            return None

    def _parse_date(self, date_elem: Optional[ET.Element]) -> datetime:
        """Parse publication date from XML element."""
        if date_elem is None:
            return datetime.now()

        try:
            year = date_elem.findtext("Year", "")
            month = date_elem.findtext("Month", "1")
            day = date_elem.findtext("Day", "1")

            if year:
                year_int = int(year)
                month_int = int(month) if month.isdigit() else 1
                day_int = int(day) if day.isdigit() else 1
                return datetime(year_int, month_int, day_int)
        except (ValueError, AttributeError):
            pass

        return datetime.now()

    def _detect_data_availability(self, abstract: str, title: str) -> str:
        """Detect data availability from abstract and title."""
        text = (abstract + " " + title).lower()

        # Check for explicit data availability statements
        if any(
            phrase in text
            for phrase in [
                "data available",
                "dataset available",
                "supplementary data",
                "data repository",
                "data deposited",
                "data sharing",
            ]
        ):
            return "available"

        if any(
            phrase in text
            for phrase in [
                "data upon request",
                "data available upon request",
                "contact author",
                "data available from",
            ]
        ):
            return "upon_request"

        if any(
            phrase in text
            for phrase in [
                "restricted access",
                "data not available",
                "confidential",
                "proprietary",
            ]
        ):
            return "restricted"

        return "unknown"

