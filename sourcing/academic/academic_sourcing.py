"""
Unified Academic Sourcing Engine for Pixelated Empathy

Consolidates multiple academic sourcing strategies:
- API-based sources (ArXiv, Semantic Scholar, CrossRef)
- Publisher integrations (APA, Springer, Oxford, etc.)
- Web scraping fallback (Google Scholar, publisher websites)

This module replaces:
- ai/pipelines/orchestrator/sourcing/academic_sourcing.py
- ai/scripts/acquire_academic_psychology_books.py
"""

import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup

# Import publisher integrations
from .publishers.apa_publisher import APAPublisher
from .publishers.base_publisher import BasePublisher

logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Academic source types"""

    # Open Access APIs
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    CROSSREF = "crossref"
    PUBMED = "pubmed"
    PUBMED_CENTRAL = "pubmed_central"
    OPENALEX = "openalex"
    CORE = "core"
    EUROPE_PMC = "europe_pmc"

    # Web Scraping
    GOOGLE_SCHOLAR = "google_scholar"
    JSTOR = "jstor"

    # Publisher Integrations
    APA_PUBLISHER = "apa_publisher"
    SPRINGER = "springer"
    SPRINGER_NATURE = "springer_nature"
    OXFORD = "oxford"
    CAMBRIDGE = "cambridge"
    WILEY = "wiley"
    ELSEVIER = "elsevier"
    TAYLOR_FRANCIS = "taylor_francis"
    SAGE = "sage"
    GUILFORD = "guilford"
    ROUTLEDGE = "routledge"


class SourcingStrategy(Enum):
    """Sourcing strategy types"""

    API_ONLY = "api_only"  # Fast, structured, limited coverage
    PUBLISHER_ONLY = "publisher_only"  # High quality, requires auth
    SCRAPING_ONLY = "scraping_only"  # Broad coverage, fragile
    HYBRID = "hybrid"  # Try all methods with fallback


@dataclass
class BookMetadata:
    """Unified book metadata structure"""

    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    isbn: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    source: str = "unknown"
    abstract: Optional[str] = None
    subject_areas: Optional[List[str]] = None
    confidence_score: float = 0.0
    therapeutic_relevance_score: Optional[float] = None
    stage_assignment: Optional[str] = None
    validation_status: str = "sourced_external"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class AcademicSourcingEngine:
    """
    Unified engine for sourcing psychology and therapy books/papers
    from academic sources.

    Features:
    - Multiple API sources (ArXiv, Semantic Scholar, CrossRef)
    - Publisher integrations (APA, Springer, Oxford, etc.)
    - Web scraping fallback (Google Scholar)
    - Intelligent fallback strategy
    - Deduplication
    - Therapeutic relevance scoring
    """

    def __init__(
        self,
        output_base_path: Optional[str] = None,
        strategy: SourcingStrategy = SourcingStrategy.HYBRID,
    ):
        self.output_base_path = Path(output_base_path or "ai/training/ready_packages/datasets")
        self.academic_literature_path = (
            self.output_base_path / "stage2_reasoning" / "academic_literature"
        )
        self._ensure_directories()
        self.strategy = strategy

        # API Configuration
        self.semantic_scholar_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.pubmed_api_key = os.getenv("PUBMED_API_KEY")
        self.openalex_api_key = os.getenv("OPENALEX_API_KEY")
        self.core_api_key = os.getenv("CORE_API_KEY")

        # API Endpoints
        self.semantic_scholar_endpoint = (
            "https://api.semanticscholar.org/graph/v1/paper/search"
        )
        self.arxiv_endpoint = "http://export.arxiv.org/api/query"
        self.crossref_endpoint = "https://api.crossref.org/works"
        self.pubmed_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.pmc_endpoint = "https://www.ncbi.nlm.nih.gov/pmc/utils"
        self.openalex_endpoint = "https://api.openalex.org/works"
        self.core_endpoint = "https://api.core.ac.uk/v3/search/works"
        self.europe_pmc_endpoint = "https://www.ebi.ac.uk/europepmc/webservices/rest"

        # Initialize publisher integrations
        self.publishers: Dict[SourceType, BasePublisher] = {}
        self._init_publishers()

        # Web scraping configuration
        self.scraping_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        # Psychology-specific search terms
        self.therapeutic_keywords = [
            "psychology",
            "psychotherapy",
            "therapy",
            "therapeutic",
            "mental health",
            "counseling",
            "clinical psychology",
            "psychiatry",
            "trauma",
            "anxiety",
            "depression",
            "cognitive behavioral therapy",
            "cbt",
            "dialectical behavior therapy",
            "dbt",
            "mindfulness",
            "emotional regulation",
            "personality disorders",
            "crisis intervention",
            "suicide prevention",
            "addiction",
            "substance abuse",
            "eating disorders",
            "ptsd",
            "post-traumatic stress",
            "neuropsychology",
            "child psychology",
            "adolescent psychology",
            "family therapy",
            "group therapy",
            "evidence-based practice",
            "clinical techniques",
            "diagnostic",
            "dsm-5",
            "assessment",
            "intervention",
            "treatment planning",
            "ethics",
            "cultural competence",
        ]

        logger.info(
            f"Initialized AcademicSourcingEngine with strategy: {strategy.value}"
        )
        logger.info(f"Output path: {self.academic_literature_path}")

    def _ensure_directories(self):
        """Ensure output directories exist"""
        self.academic_literature_path.mkdir(parents=True, exist_ok=True)

    def _init_publishers(self):
        """Initialize publisher integrations"""
        # APA is fully implemented
        try:
            apa = APAPublisher()
            api_key = os.getenv("APA_API_KEY")
            if api_key:
                apa.authenticate(api_key)
            self.publishers[SourceType.APA_PUBLISHER] = apa
            logger.info("✅ APA Publisher integration initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize APA publisher: {e}")

        # TODO: Add other publishers as they are implemented
        # self.publishers[SourceType.SPRINGER] = SpringerPublisher()
        # self.publishers[SourceType.OXFORD] = OUPPublisher()
        # etc.

    # ==================== API-Based Sources ====================

    def fetch_arxiv_papers(
        self, query: str = "psychotherapy", limit: int = 10
    ) -> List[BookMetadata]:
        """
        Fetch papers from ArXiv API (Open Access)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching ArXiv papers for: '{query}'")

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        try:
            response = requests.get(self.arxiv_endpoint, params=params, timeout=15)

            if response.status_code != 200:
                logger.error(f"ArXiv Error: {response.status_code}")
                return []

            root = ET.fromstring(response.content)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            papers = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
                summary = entry.find("atom:summary", ns).text.strip().replace("\n", " ")
                published = entry.find("atom:published", ns).text
                link = entry.find("atom:id", ns).text

                # Extract authors
                authors = []
                for author in entry.findall("atom:author", ns):
                    name = author.find("atom:name", ns).text
                    authors.append(name)

                # Create metadata
                metadata = BookMetadata(
                    title=title,
                    authors=authors,
                    publisher="ArXiv",
                    publication_year=int(published[:4]),
                    url=link,
                    source=SourceType.ARXIV.value,
                    abstract=summary,
                    subject_areas=[query],
                    confidence_score=0.8,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} papers from ArXiv")
            return papers

        except Exception as e:
            logger.error(f"ArXiv Exception: {e}")
            return []

    def fetch_semantic_scholar(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch papers from Semantic Scholar API

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching Semantic Scholar papers for: '{query}'")

        headers = {}
        if self.semantic_scholar_api_key:
            headers["x-api-key"] = self.semantic_scholar_api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "title,authors,abstract,year,citationCount,venue,url,externalIds",
        }

        try:
            response = requests.get(
                self.semantic_scholar_endpoint,
                params=params,
                headers=headers,
                timeout=10,
            )

            if response.status_code == 429:
                logger.warning("Rate limit exceeded for Semantic Scholar API")
                return []

            if response.status_code != 200:
                logger.error(f"Semantic Scholar Error: {response.status_code}")
                return []

            data = response.json()
            papers_data = data.get("data", [])

            papers = []
            for paper in papers_data:
                # Extract authors
                authors = []
                if paper.get("authors"):
                    authors = [a.get("name", "Unknown") for a in paper["authors"]]

                # Extract DOI if available
                doi = None
                external_ids = paper.get("externalIds", {})
                if external_ids and "DOI" in external_ids:
                    doi = external_ids["DOI"]

                # Create metadata
                metadata = BookMetadata(
                    title=paper.get("title", "Unknown Title"),
                    authors=authors,
                    publisher=paper.get("venue", "Unknown"),
                    publication_year=paper.get("year", 0),
                    doi=doi,
                    url=paper.get("url"),
                    source=SourceType.SEMANTIC_SCHOLAR.value,
                    abstract=paper.get("abstract"),
                    subject_areas=[query],
                    confidence_score=0.85,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} papers from Semantic Scholar")
            return papers

        except Exception as e:
            logger.error(f"Semantic Scholar Exception: {e}")
            return []

    def fetch_crossref(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch books/papers from CrossRef API (DOI resolution)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching CrossRef works for: '{query}'")

        params = {
            "query": query,
            "rows": limit,
            "filter": "type:book,type:book-chapter",
        }

        try:
            response = requests.get(self.crossref_endpoint, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"CrossRef Error: {response.status_code}")
                return []

            data = response.json()
            items = data.get("message", {}).get("items", [])

            papers = []
            for item in items:
                # Extract authors
                authors = []
                if item.get("author"):
                    for author in item["author"]:
                        given = author.get("given", "")
                        family = author.get("family", "")
                        authors.append(f"{given} {family}".strip())

                # Extract publication year
                year = 0
                if item.get("published-print"):
                    date_parts = item["published-print"].get("date-parts", [[]])[0]
                    if date_parts:
                        year = date_parts[0]

                # Extract publisher
                publisher = item.get("publisher", "Unknown")

                # Extract DOI
                doi = item.get("DOI")

                # Create metadata
                metadata = BookMetadata(
                    title=item.get("title", ["Unknown"])[0],
                    authors=authors,
                    publisher=publisher,
                    publication_year=year,
                    doi=doi,
                    url=f"https://doi.org/{doi}" if doi else None,
                    source=SourceType.CROSSREF.value,
                    subject_areas=[query],
                    confidence_score=0.9,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} works from CrossRef")
            return papers

        except Exception as e:
            logger.error(f"CrossRef Exception: {e}")
            return []

    def fetch_pubmed(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch papers from PubMed API

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching PubMed papers for: '{query}'")

        try:
            # Step 1: Search for IDs
            search_url = f"{self.pubmed_endpoint}/esearch.fcgi"
            search_params = {
                "db": "pubmed",
                "term": (
                    f"{query} AND (psychology[MeSH] OR "
                    "therapy[MeSH] OR mental health[MeSH])"
                ),
                "retmax": limit,
                "retmode": "json",
            }

            if self.pubmed_api_key:
                search_params["api_key"] = self.pubmed_api_key

            search_response = requests.get(search_url, params=search_params, timeout=10)

            if search_response.status_code != 200:
                logger.error(f"PubMed Search Error: {search_response.status_code}")
                return []

            search_data = search_response.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                logger.info("No PubMed results found")
                return []

            # Step 2: Fetch details for IDs
            fetch_url = f"{self.pubmed_endpoint}/esummary.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json",
            }

            if self.pubmed_api_key:
                fetch_params["api_key"] = self.pubmed_api_key

            fetch_response = requests.get(fetch_url, params=fetch_params, timeout=10)

            if fetch_response.status_code != 200:
                logger.error(f"PubMed Fetch Error: {fetch_response.status_code}")
                return []

            fetch_data = fetch_response.json()
            results = fetch_data.get("result", {})

            papers = []
            for pmid in id_list:
                if pmid not in results:
                    continue

                article = results[pmid]

                # Extract authors
                authors = []
                for author in article.get("authors", []):
                    if isinstance(author, dict):
                        name = author.get("name", "")
                        if name:
                            authors.append(name)

                # Extract publication year
                pub_date = article.get("pubdate", "")
                year = 0
                if pub_date:
                    try:
                        year = int(pub_date.split()[0])
                    except (ValueError, IndexError):
                        pass

                # Create metadata
                metadata = BookMetadata(
                    title=article.get("title", "Unknown Title"),
                    authors=authors,
                    publisher=article.get("source", "PubMed"),
                    publication_year=year,
                    doi=article.get("elocationid", "").replace("doi: ", ""),
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    source=SourceType.PUBMED.value,
                    subject_areas=[query],
                    confidence_score=0.9,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} papers from PubMed")
            return papers

        except Exception as e:
            logger.error(f"PubMed Exception: {e}")
            return []

    def fetch_openalex(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch papers from OpenAlex API (open bibliographic database)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching OpenAlex works for: '{query}'")

        params = {
            "search": query,
            "filter": "type:book|type:book-chapter,topics.domain.id:psychology",
            "per-page": limit,
        }

        headers = {"User-Agent": "PixelatedEmpathy/1.0 (mailto:contact@example.com)"}
        if self.openalex_api_key:
            headers["Authorization"] = f"Bearer {self.openalex_api_key}"

        try:
            response = requests.get(
                self.openalex_endpoint, params=params, headers=headers, timeout=10
            )

            if response.status_code != 200:
                logger.error(f"OpenAlex Error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])

            papers = []
            for work in results:
                # Extract authors
                authors = []
                for authorship in work.get("authorships", []):
                    author = authorship.get("author", {})
                    name = author.get("display_name", "")
                    if name:
                        authors.append(name)

                # Extract publication year
                year = work.get("publication_year", 0)

                # Extract DOI
                doi = work.get("doi", "")
                if doi and doi.startswith("https://doi.org/"):
                    doi = doi.replace("https://doi.org/", "")

                # Create metadata
                metadata = BookMetadata(
                    title=work.get("title", "Unknown Title"),
                    authors=authors,
                    publisher=work.get("primary_location", {})
                    .get("source", {})
                    .get("display_name", "Unknown"),
                    publication_year=year,
                    doi=doi,
                    url=work.get("id", ""),
                    source=SourceType.OPENALEX.value,
                    abstract=work.get("abstract", None),
                    subject_areas=[query],
                    confidence_score=0.85,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} works from OpenAlex")
            return papers

        except Exception as e:
            logger.error(f"OpenAlex Exception: {e}")
            return []

    def fetch_core(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch papers from CORE API (aggregator of open access research)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching CORE papers for: '{query}'")

        if not self.core_api_key:
            logger.warning("CORE API key not set, skipping")
            return []

        headers = {
            "Authorization": f"Bearer {self.core_api_key}",
            "Content-Type": "application/json",
        }

        params = {
            "q": query,
            "limit": limit,
        }

        try:
            response = requests.get(
                self.core_endpoint, params=params, headers=headers, timeout=10
            )

            if response.status_code != 200:
                logger.error(f"CORE Error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("results", [])

            papers = []
            for work in results:
                # Extract authors
                authors = []
                for author in work.get("authors", []):
                    if isinstance(author, str):
                        authors.append(author)
                    elif isinstance(author, dict):
                        name = author.get("name", "")
                        if name:
                            authors.append(name)

                # Extract year
                year = work.get("yearPublished", 0)

                # Create metadata
                metadata = BookMetadata(
                    title=work.get("title", "Unknown Title"),
                    authors=authors,
                    publisher=work.get("publisher", "Unknown"),
                    publication_year=year,
                    doi=work.get("doi", None),
                    url=work.get("downloadUrl")
                    or work.get("sourceFulltextUrls", [""])[0],
                    source=SourceType.CORE.value,
                    abstract=work.get("abstract", None),
                    subject_areas=[query],
                    confidence_score=0.8,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} papers from CORE")
            return papers

        except Exception as e:
            logger.error(f"CORE Exception: {e}")
            return []

    def fetch_europe_pmc(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Fetch papers from Europe PMC API

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Fetching Europe PMC papers for: '{query}'")

        search_url = f"{self.europe_pmc_endpoint}/search"
        params = {
            "query": f"{query} AND (MESH:psychology OR MESH:psychotherapy)",
            "format": "json",
            "pageSize": limit,
        }

        try:
            response = requests.get(search_url, params=params, timeout=10)

            if response.status_code != 200:
                logger.error(f"Europe PMC Error: {response.status_code}")
                return []

            data = response.json()
            results = data.get("resultList", {}).get("result", [])

            papers = []
            for article in results:
                # Extract authors
                authors = []
                author_string = article.get("authorString", "")
                if author_string:
                    authors = [a.strip() for a in author_string.split(",")]

                # Extract year
                year = 0
                pub_year = article.get("pubYear", "")
                if pub_year:
                    try:
                        year = int(pub_year)
                    except ValueError:
                        pass

                # Create metadata
                metadata = BookMetadata(
                    title=article.get("title", "Unknown Title"),
                    authors=authors,
                    publisher=article.get("journalTitle", "Europe PMC"),
                    publication_year=year,
                    doi=article.get("doi", None),
                    url=(
                        f"https://europepmc.org/article/{article.get('source', 'MED')}/"
                        f"{article.get('id', '')}"
                    ),
                    source=SourceType.EUROPE_PMC.value,
                    abstract=article.get("abstractText", None),
                    subject_areas=[query],
                    confidence_score=0.85,
                    validation_status="sourced_external",
                )

                # Score therapeutic relevance
                metadata.therapeutic_relevance_score = (
                    self._score_therapeutic_relevance(metadata)
                )

                papers.append(metadata)

            logger.info(f"✅ Found {len(papers)} papers from Europe PMC")
            return papers

        except Exception as e:
            logger.error(f"Europe PMC Exception: {e}")
            return []

    # ==================== Publisher Integrations ====================

    def fetch_from_publisher(
        self, publisher: SourceType, query: str, limit: int = 10
    ) -> List[BookMetadata]:
        """
        Fetch books from a specific publisher integration

        Args:
            publisher: Publisher source type
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        if publisher not in self.publishers:
            logger.warning(f"Publisher {publisher.value} not initialized")
            return []

        try:
            publisher_obj = self.publishers[publisher]

            # Search using publisher API
            results = publisher_obj.search_books(query=query, limit=limit)

            # Convert to unified BookMetadata format
            unified_results = []
            for book in results:
                metadata = BookMetadata(
                    title=book.title,
                    authors=book.authors,
                    publisher=book.publisher,
                    publication_year=book.publication_year,
                    isbn=book.isbn,
                    doi=book.doi,
                    url=book.raw_metadata.get("url") if book.raw_metadata else None,
                    source=publisher.value,
                    abstract=book.abstract,
                    subject_areas=[query],
                    confidence_score=0.95,
                    therapeutic_relevance_score=book.therapeutic_relevance_score,
                    stage_assignment=book.stage_assignment,
                    validation_status="sourced_publisher",
                )
                unified_results.append(metadata)

            logger.info(f"✅ Found {len(unified_results)} books from {publisher.value}")
            return unified_results

        except Exception as e:
            logger.error(f"Error fetching from {publisher.value}: {e}")
            return []

    # ==================== Web Scraping Fallback ====================

    def scrape_google_scholar(self, query: str, limit: int = 10) -> List[BookMetadata]:
        """
        Scrape Google Scholar for books (fallback method)

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of BookMetadata objects
        """
        logger.info(f"🔍 Scraping Google Scholar for: '{query}'")

        url = "https://scholar.google.com/scholar"
        params = {"q": query, "hl": "en", "as_sdt": "0,5", "num": limit}

        try:
            response = requests.get(
                url, params=params, headers=self.scraping_headers, timeout=10
            )

            if response.status_code != 200:
                logger.error(f"Google Scholar Error: {response.status_code}")
                return []

            soup = BeautifulSoup(response.content, "html.parser")

            papers = []
            for item in soup.select(".gs_ri")[:limit]:
                try:
                    title_elem = item.select_one(".gs_rt a")
                    if not title_elem:
                        continue

                    title = title_elem.get_text().strip()
                    url = title_elem.get("href", "")

                    # Extract author and publication info
                    author_pub = item.select_one(".gs_a")
                    author_pub_text = author_pub.get_text() if author_pub else ""

                    # Parse authors
                    authors = []
                    if " - " in author_pub_text:
                        authors_part = author_pub_text.split(" - ")[0]
                        if " and " in authors_part:
                            authors = [a.strip() for a in authors_part.split(" and ")]
                        else:
                            authors = [
                                a.strip() for a in authors_part.split(",") if a.strip()
                            ]

                    # Extract year
                    year = 0
                    for part in author_pub_text.split():
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            break

                    # Create metadata
                    metadata = BookMetadata(
                        title=title,
                        authors=authors,
                        publisher="Unknown",
                        publication_year=year,
                        url=url,
                        source=SourceType.GOOGLE_SCHOLAR.value,
                        subject_areas=[query],
                        confidence_score=0.7,
                        validation_status="sourced_scraping",
                    )

                    # Score therapeutic relevance
                    metadata.therapeutic_relevance_score = (
                        self._score_therapeutic_relevance(metadata)
                    )

                    papers.append(metadata)

                except Exception as e:
                    logger.warning(f"Error parsing Google Scholar result: {e}")
                    continue

            logger.info(f"✅ Scraped {len(papers)} results from Google Scholar")
            return papers

        except Exception as e:
            logger.error(f"Google Scholar scraping exception: {e}")
            return []

    # ==================== Unified Search Interface ====================

    def search_literature(
        self,
        query: str,
        limit: int = 10,
        strategy: Optional[SourcingStrategy] = None,
        sources: Optional[List[str]] = None,
    ) -> List[BookMetadata]:
        """
        Search for academic literature using specified strategy

        Args:
            query: Search query
            limit: Maximum number of results per source
            strategy: Sourcing strategy (defaults to engine's strategy)
            sources: Optional list of specific sources to query
                  (e.g. ['arxiv', 'apa_publisher'])

        Returns:
            List of BookMetadata objects (deduplicated)
        """
        strategy = strategy or self.strategy
        logger.info(f"🔍 Searching literature: '{query}' (strategy: {strategy.value})")

        all_results = []

        # Helper to check if a source should be queried
        def should_query(source_name: str) -> bool:
            if sources and source_name not in sources:
                return False
            return True

        if strategy in [SourcingStrategy.API_ONLY, SourcingStrategy.HYBRID]:
            # Try API sources first (fast and structured)
            if should_query(SourceType.ARXIV.value):
                all_results.extend(self.fetch_arxiv_papers(query, limit))
                time.sleep(1)  # Rate limiting

            if should_query(SourceType.SEMANTIC_SCHOLAR.value):
                all_results.extend(self.fetch_semantic_scholar(query, limit))
                time.sleep(1)

            if should_query(SourceType.CROSSREF.value):
                all_results.extend(self.fetch_crossref(query, limit))
                time.sleep(1)

            # New API sources
            if should_query(SourceType.PUBMED.value):
                all_results.extend(self.fetch_pubmed(query, limit))
                time.sleep(1)

            if should_query(SourceType.OPENALEX.value):
                all_results.extend(self.fetch_openalex(query, limit))
                time.sleep(1)

            # CORE requires API key
            if self.core_api_key and should_query(SourceType.CORE.value):
                all_results.extend(self.fetch_core(query, limit))
                time.sleep(1)

            if should_query(SourceType.EUROPE_PMC.value):
                all_results.extend(self.fetch_europe_pmc(query, limit))
                time.sleep(1)

        if strategy in [SourcingStrategy.PUBLISHER_ONLY, SourcingStrategy.HYBRID]:
            # Try publisher integrations (high quality, requires auth)
            for publisher_type in self.publishers.keys():
                if should_query(publisher_type.value):
                    results = self.fetch_from_publisher(publisher_type, query, limit)
                    all_results.extend(results)
                    time.sleep(1)

        if strategy in [SourcingStrategy.SCRAPING_ONLY, SourcingStrategy.HYBRID]:
            # Fall back to web scraping (broad coverage but fragile)
            if should_query(SourceType.GOOGLE_SCHOLAR.value):
                all_results.extend(self.scrape_google_scholar(query, limit))

        # Deduplicate results
        deduplicated = self._deduplicate_results(all_results)

        logger.info(f"✅ Found {len(deduplicated)} unique results for '{query}'")
        return deduplicated

    # ==================== Utility Methods ====================

    def _score_therapeutic_relevance(self, metadata: BookMetadata) -> float:
        """
        Score therapeutic relevance of a book/paper

        Args:
            metadata: Book metadata

        Returns:
            Relevance score (0.0 to 1.0)
        """
        score = 0.0

        # Check title
        title_lower = metadata.title.lower()
        title_matches = sum(1 for kw in self.therapeutic_keywords if kw in title_lower)
        if title_matches > 0:
            score += min(0.4, title_matches * 0.1)

        # Check abstract
        if metadata.abstract:
            abstract_lower = metadata.abstract.lower()
            abstract_matches = sum(
                1 for kw in self.therapeutic_keywords if kw in abstract_lower
            )
            if abstract_matches > 0:
                score += min(0.4, abstract_matches * 0.05)

        # Check subject areas
        if metadata.subject_areas:
            subject_matches = sum(
                1
                for subject in metadata.subject_areas
                for kw in self.therapeutic_keywords
                if kw in subject.lower()
            )
            if subject_matches > 0:
                score += min(0.2, subject_matches * 0.1)

        return min(score, 1.0)

    def _deduplicate_results(self, results: List[BookMetadata]) -> List[BookMetadata]:
        """
        Deduplicate results by title and first author

        Args:
            results: List of BookMetadata objects

        Returns:
            Deduplicated list
        """
        seen = set()
        deduplicated = []

        for result in results:
            # Create unique key
            title_key = result.title.lower().strip()
            author_key = (
                result.authors[0].lower().strip() if result.authors else "unknown"
            )
            key = (title_key, author_key)

            if key not in seen:
                seen.add(key)
                deduplicated.append(result)

        return deduplicated

    def export_data(
        self, data: List[BookMetadata], filename: str = "academic_batch_001.json"
    ) -> Path:
        """
        Export literature data to JSON file

        Args:
            data: List of BookMetadata objects
            filename: Output filename

        Returns:
            Path to exported file
        """
        output_file = self.academic_literature_path / filename

        try:
            # Convert to dictionaries
            data_dicts = [item.to_dict() for item in data]

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(data_dicts, f, indent=2, ensure_ascii=False)

            logger.info(f"💾 Exported {len(data)} records to {output_file}")
            return output_file

        except Exception as e:
            logger.error(f"Failed to export data: {e}")
            raise

    def run_sourcing_pipeline(
        self, queries: Optional[List[str]] = None, limit_per_query: int = 10
    ) -> Path:
        """
        Run complete sourcing pipeline

        Args:
            queries: List of search queries (defaults to therapeutic topics)
            limit_per_query: Maximum results per query

        Returns:
            Path to exported data file
        """
        if queries is None:
            queries = [
                "mental health therapy",
                "cognitive behavioral therapy",
                "trauma-informed care",
                "clinical psychology",
                "psychotherapy techniques",
            ]

        logger.info("🚀 Starting Academic Sourcing Pipeline...")

        all_data = []
        for query in queries:
            results = self.search_literature(query, limit=limit_per_query)
            all_data.extend(results)
            time.sleep(2)  # Be respectful with rate limiting

        # Deduplicate across all queries
        deduplicated = self._deduplicate_results(all_data)

        # Export
        output_path = self.export_data(deduplicated)

        logger.info("✅ Academic Sourcing Pipeline Complete!")
        logger.info(f"   Total unique results: {len(deduplicated)}")
        logger.info(f"   Output: {output_path}")

        return output_path


# Convenience function for backward compatibility
def create_academic_sourcing_engine(
    output_path: Optional[str] = None, strategy: str = "hybrid"
) -> AcademicSourcingEngine:
    """
    Create AcademicSourcingEngine with specified configuration

    Args:
        output_path: Output directory path
        strategy: Sourcing strategy
            ('api_only', 'publisher_only', 'scraping_only', 'hybrid')

    Returns:
        Configured AcademicSourcingEngine instance
    """
    strategy_enum = SourcingStrategy[strategy.upper()]
    return AcademicSourcingEngine(output_base_path=output_path, strategy=strategy_enum)
