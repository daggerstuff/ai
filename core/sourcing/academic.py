#!/usr/bin/env python3
"""Academic Sourcing Module for Mental Health AI Training Data

This module provides production-quality functionality for sourcing academic content
from multiple databases including PubMed, arXiv, and Semantic Scholar. It handles
search, metadata extraction, content parsing, and mental health-specific filtering.

Key Features:
- Multi-source academic database integration (PubMed, arXiv, Semantic Scholar)
- DOI resolution and metadata extraction
- PDF processing and content extraction
- Rate limiting and API quota management
- Result caching to avoid redundant API calls
- Mental health terminology detection and filtering
- Structured JSON output matching PIX-32 schema
- Citation graph support

Usage:
    from ai.core.sourcing.academic import AcademicSourcing

    sourcing = AcademicSourcing()
    results = sourcing.search(
        keywords="cognitive behavioral therapy depression",
        sources=["pubmed", "semantic_scholar"],
        max_results=50
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
import re
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import aiohttp
import pdfplumber
from pydantic import BaseModel, Field, HttpUrl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("academic_sourcing")

# ============================================================================
# Configuration and Data Classes
# ============================================================================


class SourceType(StrEnum):
    """Supported academic data sources."""

    PUBMED = "pubmed"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    DOI = "doi"
    LOCAL_PDF = "local_pdf"


class StudyType(StrEnum):
    """Types of academic studies."""

    RCT = "randomized_controlled_trial"
    CASE_STUDY = "case_study"
    REVIEW = "review"
    META_ANALYSIS = "meta_analysis"
    SYSTEMATIC_REVIEW = "systematic_review"
    OBSERVATIONAL = "observational"
    LONGITUDINAL = "longitudinal"
    CROSS_SECTIONAL = "cross_sectional"
    QUALITATIVE = "qualitative"
    MIXED_METHODS = "mixed_methods"
    UNKNOWN = "unknown"


class AccessStatus(StrEnum):
    """Access status for papers."""

    OPEN_ACCESS = "open_access"
    PAYWALLED = "paywalled"
    ABSTRACT_ONLY = "abstract_only"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limiting configuration for API calls."""

    requests_per_second: float = 3.0
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    burst_size: int = 10

    def get_delay(self) -> float:
        """Calculate delay between requests."""
        return 1.0 / self.requests_per_second


@dataclass(frozen=True)
class CacheConfig:
    """Caching configuration."""

    enabled: bool = True
    ttl_seconds: int = 3600  # 1 hour default
    max_size: int = 10000
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "academic_sourcing")


@dataclass
class AcademicSourcingConfig:
    """Configuration for academic sourcing operations."""

    # Concurrency limits
    pdf_concurrency: int = 8
    # API endpoints
    pubmed_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    semantic_scholar_base_url: str = "https://api.semanticscholar.org/graph/v1"
    doi_resolver_url: str = "https://doi.org"

    # Rate limiting
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)

    # Caching
    cache: CacheConfig = field(default_factory=CacheConfig)

    # Search parameters
    default_max_results: int = 50
    default_retmax: int = 20
    default_sort: str = "relevance"

    # Content extraction
    extract_full_text: bool = True
    extract_key_findings: bool = True
    extract_methodology: bool = True

    # Mental health filtering
    enable_mental_health_filter: bool = True
    mental_health_keywords: list[str] = field(
        default_factory=lambda: [
            "depression",
            "anxiety",
            "bipolar",
            "schizophrenia",
            "ptsd",
            "trauma",
            "therapy",
            "counseling",
            "psychiatry",
            "psychology",
            "mental health",
            "cognitive behavioral",
            "cbt",
            "psychotherapy",
            "mood disorder",
            "anxiety disorder",
            "personality disorder",
            "eating disorder",
            "substance abuse",
            "addiction",
            "self-harm",
            "suicide",
            "crisis",
            "intervention",
            "treatment",
            "diagnosis",
            "assessment",
            "screening",
            "evidence-based",
            "clinical",
            "therapeutic",
        ]
    )

    # API keys (optional)
    semantic_scholar_api_key: str | None = None

    # Request timeout
    request_timeout: int = 30

    # Retry configuration
    max_retries: int = 3
    retry_backoff: float = 1.0


@dataclass
class Author:
    """Author information."""

    name: str
    affiliation: str | None = None
    orcid: str | None = None


@dataclass
class Citation:
    """Citation information."""

    paper_id: str
    title: str
    year: int | None = None
    venue: str | None = None


@dataclass
class PaperMetadata:
    """Metadata for an academic paper."""

    # Identifiers
    paper_id: str
    doi: str | None = None
    pmid: str | None = None
    arxiv_id: str | None = None

    # Basic info
    title: str
    authors: list[Author] = field(default_factory=list)
    publication_date: datetime | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None

    # Content
    abstract: str | None = None
    full_text: str | None = None
    keywords: list[str] = field(default_factory=list)

    # Study info
    study_type: StudyType = StudyType.UNKNOWN
    access_status: AccessStatus = AccessStatus.UNKNOWN

    # URLs
    pdf_url: str | None = None
    html_url: str | None = None

    # Metrics
    citation_count: int = 0
    references: list[Citation] = field(default_factory=list)
    cited_by: list[Citation] = field(default_factory=list)

    # Mental health relevance
    mental_health_relevance_score: float = 0.0
    mental_health_topics: list[str] = field(default_factory=list)

    # Provenance
    source: SourceType = SourceType.PUBMED
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_data: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================


class AuthorModel(BaseModel):
    """Pydantic model for author information."""

    name: str = Field(..., description="Full name of the author")
    affiliation: str | None = Field(None, description="Institutional affiliation")
    orcid: str | None = Field(None, description="ORCID identifier")


class CitationModel(BaseModel):
    """Pydantic model for citation information."""

    paper_id: str = Field(..., description="Unique identifier for the cited paper")
    title: str = Field(..., description="Title of the cited paper")
    year: int | None = Field(None, description="Publication year")
    venue: str | None = Field(None, description="Publication venue")


class PaperMetadataModel(BaseModel):
    """Pydantic model for paper metadata (PIX-32 compatible)."""

    # Identifiers
    paper_id: str = Field(..., description="Unique identifier for the paper")
    doi: str | None = Field(None, description="Digital Object Identifier")
    pmid: str | None = Field(None, description="PubMed ID")
    arxiv_id: str | None = Field(None, description="arXiv ID")

    # Basic information
    title: str = Field(..., description="Paper title")
    authors: list[AuthorModel] = Field(default_factory=list, description="List of authors")
    publication_date: str | None = Field(None, description="ISO format publication date")
    journal: str | None = Field(None, description="Journal name")
    volume: str | None = Field(None, description="Volume number")
    issue: str | None = Field(None, description="Issue number")
    pages: str | None = Field(None, description="Page numbers")

    # Content
    abstract: str | None = Field(None, description="Paper abstract")
    full_text: str | None = Field(None, description="Full text content")
    keywords: list[str] = Field(default_factory=list, description="Paper keywords")

    # Study information
    study_type: str = Field(default="unknown", description="Type of study")
    access_status: str = Field(default="unknown", description="Access status")

    # URLs
    pdf_url: HttpUrl | None = Field(None, description="URL to PDF")
    html_url: HttpUrl | None = Field(None, description="URL to HTML version")

    # Metrics
    citation_count: int = Field(default=0, description="Number of citations")
    references: list[CitationModel] = Field(default_factory=list, description="References")
    cited_by: list[CitationModel] = Field(default_factory=list, description="Citations")

    # Mental health relevance
    mental_health_relevance_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Relevance score for mental health"
    )
    mental_health_topics: list[str] = Field(default_factory=list, description="Detected mental health topics")

    # Provenance
    source: str = Field(default="pubmed", description="Data source")
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="Retrieval timestamp (ISO format)",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
        }


# ============================================================================
# Cache Implementation
# ============================================================================


class CacheEntry(BaseModel):
    """Cache entry model."""

    key: str
    data: dict[str, Any]
    timestamp: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return (datetime.now(UTC) - self.timestamp).total_seconds() > self.ttl_seconds


class SimpleCache:
    """Simple in-memory cache with optional disk persistence."""

    def __init__(self, config: CacheConfig):
        """Initialize cache with configuration."""
        self.config = config
        self._cache: dict[str, CacheEntry] = {}
        self._cache_dir = config.cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if config.enabled:
            self._load_from_disk()

    def _get_cache_path(self) -> Path:
        """Get path to cache file."""
        return self._cache_dir / "academic_cache.json"

    def _load_from_disk(self) -> None:
        """Load cache from disk."""
        cache_path = self._get_cache_path()
        if not cache_path.exists():
            return

        try:
            with open(cache_path) as f:
                data = json.load(f)
            for key, entry_data in data.items():
                entry = CacheEntry(
                    key=key,
                    data=entry_data["data"],
                    timestamp=datetime.fromisoformat(entry_data["timestamp"]),
                    ttl_seconds=entry_data["ttl_seconds"],
                )
                if not entry.is_expired():
                    self._cache[key] = entry
            logger.info(f"Loaded {len(self._cache)} entries from cache")
        except Exception as e:
            logger.warning(f"Failed to load cache from disk: {e}")

    def _save_to_disk(self) -> None:
        """Save cache to disk."""
        if not self.config.enabled:
            return
        cache_path = self._get_cache_path()
        try:
            data = {
                key: {
                    "data": entry.data,
                    "timestamp": entry.timestamp.isoformat(),
                    "ttl_seconds": entry.ttl_seconds,
                }
                for key, entry in self._cache.items()
            }
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache to disk: {e}")

    def _generate_key(self, source: str, query: str, params: dict[str, Any]) -> str:
        """Generate cache key from parameters."""
        key_data = f"{source}:{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, source: str, query: str, params: dict[str, Any]) -> dict[str, Any] | None:
        """Get cached data if available and not expired."""
        if not self.config.enabled:
            return None
        key = self._generate_key(source, query, params)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if entry.is_expired():
            del self._cache[key]
            return None
        logger.debug(f"Cache hit for key: {key[:16]}...")
        return entry.data

    def set(
        self,
        source: str,
        query: str,
        params: dict[str, Any],
        data: dict[str, Any],
    ) -> None:
        """Cache data with TTL."""
        if not self.config.enabled:
            return
        # Enforce max size
        if len(self._cache) >= self.config.max_size:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
            del self._cache[oldest_key]

        key = self._generate_key(source, query, params)
        entry = CacheEntry(
            key=key,
            data=data,
            timestamp=datetime.now(UTC),
            ttl_seconds=self.config.ttl_seconds,
        )
        self._cache[key] = entry
        # Periodically save to disk
        if len(self._cache) % 10 == 0:
            self._save_to_disk()

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._save_to_disk()
        logger.info("Cache cleared")


# ============================================================================
# Rate Limiter
# ============================================================================


class RateLimiter:
    """Rate limiter using token bucket algorithm."""

    def __init__(self, config: RateLimitConfig):
        """Initialize rate limiter with configuration."""
        self.config = config
        self._tokens = config.burst_size
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            # Refill tokens based on elapsed time
            refill = elapsed * self.config.requests_per_second
            self._tokens = min(self.config.burst_size, self._tokens + refill)
            self._last_update = now

            if self._tokens < 1:
                # Calculate wait time
                wait_time = (1 - self._tokens) / self.config.requests_per_second
                logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                self._tokens = 1
            else:
                self._tokens -= 1


# ============================================================================
# API Client Base
# ============================================================================


class AcademicAPIClient:
    """Base client for academic API interactions."""

    def __init__(self, config: AcademicSourcingConfig):
        """Initialize API client with configuration."""
        self.config = config
        self.cache = SimpleCache(config.cache)
        self.rate_limiter = RateLimiter(config.rate_limit)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.config.request_timeout))
        return self._session

    async def _request_with_retry(
        self,
        method: str,
        url: str,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        cache_key: str | None = None,
    ) -> dict[str, Any]:
        """Make HTTP request with retry logic and caching."""

        # Check cache first
        if cache_key and (cached := self.cache.get(cache_key[0], cache_key[1], cache_key[2])):
            return cached

        # Rate limit
        await self.rate_limiter.acquire()

        session = await self._get_session()
        for attempt in range(self.config.max_retries):
            try:
                async with session.request(method, url, params=params, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Cache the result
                    if cache_key:
                        self.cache.set(cache_key[0], cache_key[1], cache_key[2], data)
                    return data
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise

                wait_time = self.config.retry_backoff * (2**attempt)
                logger.warning(
                    "Request failed (attempt %s), retrying in %ss: %s",
                    attempt + 1,
                    wait_time,
                    e,
                )
                await asyncio.sleep(wait_time)

        raise RuntimeError("Max retries exceeded")

    def _extract_mental_health_topics(self, text: str | None, keywords: list[str]) -> list[str]:
        """Extract mental health topics from text."""
        if not text:
            return []
        text_lower = text.lower()
        return [keyword for keyword in keywords if keyword.lower() in text_lower]

    def _calculate_mental_health_relevance(self, topics: list[str], total_keywords: int) -> float:
        """Calculate mental health relevance score."""
        # Simple relevance based on keyword matches
        # Could be enhanced with more sophisticated NLP
        return min(1.0, len(topics) / max(1, total_keywords * 0.1)) if topics else 0.0


# ============================================================================
# PubMed API Client
# ============================================================================


class PubMedClient(AcademicAPIClient):
    """Client for PubMed API interactions."""

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort: str = "relevance",
    ) -> list[PaperMetadata]:
        """Search PubMed for papers matching query."""
        logger.info(f"Searching PubMed for: {query}")

        # Build ESearch query
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "sort": sort,
            "retmode": "json",
        }
        cache_key = ("pubmed", "search", search_params)

        search_data = await self._request_with_retry(
            "GET",
            f"{self.config.pubmed_base_url}/esearch.fcgi",
            params=search_params,
            cache_key=cache_key,
        )

        pmids = search_data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            logger.info(f"No PubMed results found for: {query}")
            return []

        logger.info(f"Found {len(pmids)} PubMed IDs, fetching details...")

        # Fetch details for each PMID
        # Performance optimization: use asyncio.gather to fetch details concurrently
        # Limit the number of concurrent PubMed detail fetches to avoid overloading the API
        max_concurrent_fetches = 10
        semaphore = asyncio.Semaphore(max_concurrent_fetches)

        async def fetch_and_catch(pmid):
            async with semaphore:
                try:
                    return await self.fetch_details(pmid)
                except Exception as e:
                    logger.warning(f"Failed to fetch details for PMID {pmid}: {e}")
                    return None

        results = await asyncio.gather(*(fetch_and_catch(pmid) for pmid in pmids))
        papers = [p for p in results if p is not None]

        logger.info(f"Successfully retrieved {len(papers)} papers from PubMed")
        return papers

    async def fetch_details(self, pmid: str) -> PaperMetadata | None:
        """Fetch detailed metadata for a specific PubMed ID."""
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "retmode": "json",
        }
        cache_key = ("pubmed", "fetch_details", fetch_params)

        data = await self._request_with_retry(
            "GET",
            f"{self.config.pubmed_base_url}/efetch.fcgi",
            params=fetch_params,
            cache_key=cache_key,
        )

        # Parse PubMed data
        try:
            result = data.get("result", {})
            if not result or pmid not in result:
                return None

            pubmed_data = result[pmid]
            article = pubmed_data.get("article", {})
            medline_citation = pubmed_data.get("medlinecitation", {})

            # Extract basic info
            title = self._extract_title(article)
            authors = self._extract_authors(article)
            abstract = self._extract_abstract(article)
            journal = self._extract_journal(article)
            publication_date = self._extract_publication_date(pubmed_data)
            keywords = self._extract_keywords(medline_citation)

            # Extract study type from publication type
            study_type = self._detect_study_type(medline_citation)

            # Mental health filtering
            topics = self._extract_mental_health_topics(f"{title} {abstract or ''}", self.config.mental_health_keywords)
            relevance_score = self._calculate_mental_health_relevance(topics, len(self.config.mental_health_keywords))

            # Determine access status
            access_status = self._determine_access_status(pubmed_data)

            return PaperMetadata(
                paper_id=f"pubmed:{pmid}",
                pmid=pmid,
                title=title,
                authors=authors,
                abstract=abstract,
                journal=journal,
                publication_date=publication_date,
                keywords=keywords,
                study_type=study_type,
                access_status=access_status,
                html_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}",
                mental_health_relevance_score=relevance_score,
                mental_health_topics=topics,
                source=SourceType.PUBMED,
                raw_data=pubmed_data,
            )
        except Exception as e:
            logger.warning(f"Failed to parse PubMed data for {pmid}: {e}")
            return None

    def _extract_title(self, article: dict[str, Any]) -> str:
        """Extract title from article data."""
        title_list = article.get("title", [])
        return title_list[0] if title_list else ""

    def _extract_authors(self, article: dict[str, Any]) -> list[Author]:
        """Extract authors from article data."""
        authors = []
        author_list = article.get("authors", [])
        for author_data in author_list:
            if isinstance(author_data, dict):
                author = Author(
                    name=author_data.get("name", ""),
                    affiliation=author_data.get("affiliation"),
                )
                authors.append(author)
        return authors

    def _extract_abstract(self, article: dict[str, Any]) -> str | None:
        """Extract abstract from article data."""
        abstract_data = article.get("abstract", {})
        if not abstract_data:
            return None
        abstract_text = abstract_data.get("abstracttext", "")
        if isinstance(abstract_text, list):
            abstract_text = " ".join(abstract_text)
        return abstract_text.strip() if abstract_text else None

    def _extract_journal(self, article: dict[str, Any]) -> str | None:
        """Extract journal name from article data."""
        journal = article.get("journal", {})
        return journal.get("name") if journal else None

    def _extract_publication_date(self, pubmed_data: dict[str, Any]) -> datetime | None:
        """Extract publication date from PubMed data."""
        with suppress(Exception):
            medline_citation = pubmed_data.get("medlinecitation", {})
            article = medline_citation.get("article", {})
            journal = article.get("journal", {})
            journal_issue = journal.get("journalissue", {})

            if pub_date := journal_issue.get("pubdate", {}):
                year = pub_date.get("year")
                month = pub_date.get("month")
                day = pub_date.get("day")
                if year:
                    month_str = f"{month:02d}" if month else "01"
                    day_str = f"{day:02d}" if day else "01"
                    return datetime.strptime(f"{year}-{month_str}-{day_str}", "%Y-%m-%d")
        return None

    def _extract_keywords(self, medline_citation: dict[str, Any]) -> list[str]:
        """Extract keywords from MedLine citation."""
        keywords = []
        keyword_list = medline_citation.get("keywordlist", [])
        if isinstance(keyword_list, dict):
            for keyword in keyword_list.get("keyword", []):
                if isinstance(keyword, dict):
                    keywords.append(keyword.get("#text", ""))
                elif isinstance(keyword, str):
                    keywords.append(keyword)
        return keywords

    def _detect_study_type(self, medline_citation: dict[str, Any]) -> StudyType:
        """Detect study type from publication type and keywords."""
        publication_types = medline_citation.get("article", {}).get("publicationtypelist", [])
        if not publication_types:
            return StudyType.UNKNOWN

        pub_types = []
        for pub_type in publication_types:
            if isinstance(pub_type, dict):
                pub_types.append(pub_type.get("#text", "").lower())
            elif isinstance(pub_type, str):
                pub_types.append(pub_type.lower())

        # Detect study type from publication types
        type_str = " ".join(pub_types)
        if "randomized" in type_str or "clinical trial" in type_str:
            return StudyType.RCT
        if "review" in type_str:
            return StudyType.REVIEW
        if "meta-analysis" in type_str:
            return StudyType.META_ANALYSIS
        if "case report" in type_str or "case study" in type_str:
            return StudyType.CASE_STUDY
        if "observational" in type_str:
            return StudyType.OBSERVATIONAL
        return StudyType.UNKNOWN

    def _determine_access_status(self, pubmed_data: dict[str, Any]) -> AccessStatus:
        """Determine access status from PubMed data."""
        # PubMed Central availability
        pmc_data = pubmed_data.get("pmc", {})
        if pmc_data and pmc_data.get("pmc"):
            return AccessStatus.OPEN_ACCESS

        # Check if free article
        article = pubmed_data.get("medlinecitation", {}).get("article", {})
        if article.get("isfreearticle"):
            return AccessStatus.OPEN_ACCESS

        return AccessStatus.PAYWALLED


# ============================================================================
# arXiv API Client
# ============================================================================


class ArXivClient(AcademicAPIClient):
    """Client for arXiv API interactions."""

    async def search(
        self,
        query: str,
        max_results: int = 20,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[PaperMetadata]:
        """Search arXiv for papers matching query."""
        logger.info(f"Searching arXiv for: {query}")

        # Build arXiv query
        search_params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        cache_key = ("arxiv", "search", search_params)

        data = await self._request_with_retry(
            "GET",
            self.config.arxiv_base_url,
            params=search_params,
            cache_key=cache_key,
        )

        # Parse arXiv data
        papers = []
        entries = data.get("feed", {}).get("entry", [])
        for entry in entries:
            try:
                if paper := self._parse_entry(entry):
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse arXiv entry: {e}")

        logger.info(f"Successfully retrieved {len(papers)} papers from arXiv")
        return papers

    def _parse_entry(self, entry: dict[str, Any]) -> PaperMetadata | None:
        """Parse arXiv entry into PaperMetadata."""
        try:
            # Extract basic info
            arxiv_id = entry.get("id", "").split("/")[-1]
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            published = entry.get("published", "")

            # Parse authors
            authors = [
                Author(name=author_data.get("name", ""))
                for author_data in entry.get("authors", [])
                if isinstance(author_data, dict)
            ]

            # Parse publication date
            publication_date = None
            if published:
                with suppress(ValueError):
                    publication_date = datetime.fromisoformat(published.replace("Z", "+00:00"))

            # Extract categories as keywords
            categories = [
                category.get("term", "") for category in entry.get("category", []) if isinstance(category, dict)
            ]

            # Find PDF link
            pdf_url = next(
                (
                    link.get("href")
                    for link in entry.get("link", [])
                    if isinstance(link, dict) and link.get("type") == "application/pdf"
                ),
                None,
            )

            # Mental health filtering
            topics = self._extract_mental_health_topics(f"{title} {summary}", self.config.mental_health_keywords)
            relevance_score = self._calculate_mental_health_relevance(topics, len(self.config.mental_health_keywords))

            return PaperMetadata(
                paper_id=f"arxiv:{arxiv_id}",
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=summary,
                publication_date=publication_date,
                keywords=categories,
                access_status=AccessStatus.OPEN_ACCESS,
                pdf_url=pdf_url,
                html_url=f"https://arxiv.org/abs/{arxiv_id}",
                mental_health_relevance_score=relevance_score,
                mental_health_topics=topics,
                source=SourceType.ARXIV,
                raw_data=entry,
            )
        except Exception as e:
            logger.warning(f"Failed to parse arXiv entry: {e}")
            return None


# ============================================================================
# Semantic Scholar API Client
# ============================================================================


class SemanticScholarClient(AcademicAPIClient):
    """Client for Semantic Scholar API interactions."""

    async def search(
        self,
        query: str,
        max_results: int = 20,
        fields: list[str] | None = None,
    ) -> list[PaperMetadata]:
        """Search Semantic Scholar for papers matching query."""
        logger.info(f"Searching Semantic Scholar for: {query}")

        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "authors",
                "year",
                "venue",
                "url",
                "isOpenAccess",
                "openAccessPdf",
                "citationCount",
                "influentialCitationCount",
                "publicationDate",
                "fieldsOfStudy",
            ]

        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key

        # Build request body for search
        search_body = {
            "query": query,
            "limit": max_results,
            "fields": fields,
        }

        cache_key = (
            "semantic_scholar",
            "search",
            {"query": query, "max_results": max_results},
        )

        data = await self._request_with_retry(
            "POST",
            f"{self.config.semantic_scholar_base_url}/paper/search",
            json=search_body,
            headers=headers,
            cache_key=cache_key,
        )

        # Parse Semantic Scholar data
        papers = []
        entries = data.get("data", [])
        for entry in entries:
            try:
                if paper := self._parse_entry(entry):
                    papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse Semantic Scholar entry: {e}")

        logger.info(f"Successfully retrieved {len(papers)} papers from Semantic Scholar")
        return papers

    async def get_paper_details(self, paper_id: str, fields: list[str] | None = None) -> PaperMetadata | None:
        """Get detailed information for a specific paper."""
        logger.info(f"Fetching details for paper: {paper_id}")

        if fields is None:
            fields = [
                "paperId",
                "title",
                "abstract",
                "authors",
                "year",
                "venue",
                "url",
                "isOpenAccess",
                "openAccessPdf",
                "citationCount",
                "influentialCitationCount",
                "publicationDate",
                "fieldsOfStudy",
                "references",
                "citations",
            ]

        headers = {}
        if self.config.semantic_scholar_api_key:
            headers["x-api-key"] = self.config.semantic_scholar_api_key

        cache_key = ("semantic_scholar", "get_paper_details", {"paper_id": paper_id})

        data = await self._request_with_retry(
            "GET",
            f"{self.config.semantic_scholar_base_url}/paper/{paper_id}",
            params={"fields": ",".join(fields)},
            headers=headers,
            cache_key=cache_key,
        )

        paper_data = data
        return self._parse_entry(paper_data) if paper_data else None

    def _parse_entry(self, entry: dict[str, Any]) -> PaperMetadata | None:
        """Parse Semantic Scholar entry into PaperMetadata."""
        try:
            # Extract basic info
            paper_id = entry.get("paperId", "")
            title = entry.get("title", "")
            abstract = entry.get("abstract")
            venue = entry.get("venue")
            url = entry.get("url")
            is_open_access = entry.get("isOpenAccess", False)
            citation_count = entry.get("citationCount", 0)
            publication_date_str = entry.get("publicationDate")
            fields_of_study = entry.get("fieldsOfStudy", [])

            # Parse authors
            authors = [
                Author(name=author_data.get("name", ""))
                for author_data in entry.get("authors", [])
                if isinstance(author_data, dict)
            ]

            # Parse publication date
            publication_date = None
            if publication_date_str:
                with suppress(ValueError):
                    publication_date = datetime.fromisoformat(publication_date_str)

            # Parse PDF URL
            pdf_url = None
            open_access_pdf = entry.get("openAccessPdf")
            if open_access_pdf and isinstance(open_access_pdf, dict):
                pdf_url = open_access_pdf.get("url")

            # Parse references and citations
            references = [
                Citation(
                    paper_id=ref.get("paperId", ""),
                    title=ref.get("title", ""),
                    year=ref.get("year"),
                    venue=ref.get("venue"),
                )
                for ref in entry.get("references", [])
                if isinstance(ref, dict)
            ]
            cited_by = [
                Citation(
                    paper_id=cit.get("paperId", ""),
                    title=cit.get("title", ""),
                    year=cit.get("year"),
                    venue=cit.get("venue"),
                )
                for cit in entry.get("citations", [])
                if isinstance(cit, dict)
            ]

            # Mental health filtering
            topics = self._extract_mental_health_topics(f"{title} {abstract or ''}", self.config.mental_health_keywords)
            relevance_score = self._calculate_mental_health_relevance(topics, len(self.config.mental_health_keywords))

            # Determine access status
            access_status = AccessStatus.OPEN_ACCESS if is_open_access else AccessStatus.PAYWALLED

            if not abstract and not pdf_url:
                access_status = AccessStatus.ABSTRACT_ONLY

            return PaperMetadata(
                paper_id=f"semantic_scholar:{paper_id}",
                title=title,
                authors=authors,
                abstract=abstract,
                publication_date=publication_date,
                journal=venue,
                citation_count=citation_count,
                references=references,
                cited_by=cited_by,
                keywords=fields_of_study or [],
                access_status=access_status,
                pdf_url=pdf_url,
                html_url=url,
                mental_health_relevance_score=relevance_score,
                mental_health_topics=topics,
                source=SourceType.SEMANTIC_SCHOLAR,
                raw_data=entry,
            )
        except Exception as e:
            logger.warning(f"Failed to parse Semantic Scholar entry: {e}")
            return None


# ============================================================================
# DOI Resolver
# ============================================================================


class DOIResolver(AcademicAPIClient):
    """Client for DOI resolution and metadata extraction."""

    async def resolve(self, doi: str) -> PaperMetadata | None:
        """Resolve DOI and extract metadata."""
        logger.info(f"Resolving DOI: {doi}")

        # Clean DOI
        doi = doi.strip().lower()
        if not doi.startswith("10."):
            logger.warning(f"Invalid DOI format: {doi}")
            return None

        # Use Content Negotiation to get metadata
        headers = {
            "Accept": "application/vnd.citationstyles.csl+json",
        }
        cache_key = ("doi", "resolve", {"doi": doi})

        try:
            data = await self._request_with_retry(
                "GET",
                f"{self.config.doi_resolver_url}/{doi}",
                headers=headers,
                cache_key=cache_key,
            )

            # Parse metadata
            return self._parse_metadata(doi, data)
        except Exception as e:
            logger.warning(f"Failed to resolve DOI {doi}: {e}")
            return None

    def _parse_metadata(self, doi: str, data: dict[str, Any]) -> PaperMetadata | None:
        """Parse DOI metadata into PaperMetadata."""
        try:
            title = data.get("title", "")

            authors = [
                Author(name=author.get("family", "") + " " + author.get("given", ""))
                for author in data.get("author", [])
                if isinstance(author, dict)
            ]

            # Parse publication date
            publication_date = None
            issued = data.get("issued")
            if issued and isinstance(issued, dict):
                date_parts = issued.get("date-parts", [])
                if date_parts and isinstance(date_parts[0], list):
                    year, month, day = date_parts[0]
                    month = month or 1
                    day = day or 1
                    publication_date = datetime(year, month, day)

            # Extract other metadata
            journal = data.get("container-title", "")
            if isinstance(journal, list) and journal:
                journal = journal[0]
            volume = data.get("volume")
            issue = data.get("issue")
            pages = data.get("page")

            # Mental health filtering
            topics = self._extract_mental_health_topics(
                f"{title} {data.get('abstract', '')}",
                self.config.mental_health_keywords,
            )
            relevance_score = self._calculate_mental_health_relevance(topics, len(self.config.mental_health_keywords))

            return PaperMetadata(
                paper_id=f"doi:{doi}",
                doi=doi,
                title=title,
                authors=authors,
                publication_date=publication_date,
                journal=journal,
                volume=volume,
                issue=issue,
                pages=pages,
                access_status=AccessStatus.UNKNOWN,
                html_url=f"https://doi.org/{doi}",
                mental_health_relevance_score=relevance_score,
                mental_health_topics=topics,
                source=SourceType.DOI,
                raw_data=data,
            )
        except Exception as e:
            logger.warning(f"Failed to parse DOI metadata: {e}")
            return None


# ============================================================================
# PDF Processor
# ============================================================================


class PDFProcessor:
    """Processor for extracting content from PDF files."""

    def __init__(self, config: AcademicSourcingConfig):
        """Initialize PDF processor with configuration."""
        self.config = config
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create shared HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        """Close the shared HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def extract_text(self, pdf_url: str) -> str | None:
        """Extract text from PDF URL."""
        logger.info(f"Extracting text from PDF: {pdf_url}")

        try:
            # Download PDF using shared session
            session = await self._get_session()
            async with session.get(pdf_url) as response:
                response.raise_for_status()
                pdf_content = await response.read()

            # Extract text using pdfplumber if available
            try:
                pdf_file = io.BytesIO(pdf_content)
                text_parts = []
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        if text := page.extract_text():
                            text_parts.append(text)
                full_text = "\n\n".join(text_parts)
                logger.info(f"Extracted {len(full_text)} characters from PDF")
                return full_text
            except ImportError:
                logger.warning("pdfplumber not available, skipping PDF extraction")
                return None
        except Exception as e:
            logger.warning(f"Failed to extract text from PDF: {e}")
            return None

    def extract_key_findings(self, text: str) -> list[str]:
        """Extract key findings from text."""
        findings = []

        # Look for common patterns in academic papers
        patterns = [
            r"(?:results|findings|conclusion)s?:\s*([^.!?]+[.!?])",
            r"(?:we found|our results show|this study demonstrates)\s*([^.!?]+[.!?])",
            r"(?:significant|notable)\s+(?:finding|result|difference)\s*([^.!?]+[.!?])",
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                finding = match[1].strip()
                if len(finding) > 20:  # Filter out very short matches
                    findings.append(finding)

        return findings[:10]  # Limit to top 10 findings

    def extract_methodology(self, text: str) -> str | None:
        """Extract methodology section from text."""
        # Look for methodology section
        methodology_patterns = [
            r"(?:methodology|methods|procedure|design)\s*[:]\s*(.*?)(?=\n\n(?:introduction|results|discussion|conclusion)s?\s*[:])",
            r"(?:methods|procedure)\s*[:]\s*(.*?)(?=\n\n(?:results|discussion|conclusion)s?\s*[:])",
        ]

        for pattern in methodology_patterns:
            if match := re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                methodology = match[1].strip()
                if len(methodology) > 100:  # Ensure we got substantial content
                    return methodology[:2000]  # Limit length
        return None


# ============================================================================
# Main Academic Sourcing Class
# ============================================================================


class AcademicSourcing:
    """Main class for academic sourcing operations."""

    def __init__(self, config: AcademicSourcingConfig | None = None):
        """Initialize academic sourcing with configuration."""
        self.config = config or AcademicSourcingConfig()

        self._pubmed_client: PubMedClient | None = None
        self._arxiv_client: ArXivClient | None = None
        self._semantic_scholar_client: SemanticScholarClient | None = None
        self._doi_resolver: DOIResolver | None = None
        self._pdf_processor: PDFProcessor | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._pubmed_client = PubMedClient(self.config)
        self._arxiv_client = ArXivClient(self.config)
        self._semantic_scholar_client = SemanticScholarClient(self.config)
        self._doi_resolver = DOIResolver(self.config)
        self._pdf_processor = PDFProcessor(self.config)

        await self._pubmed_client.__aenter__()
        await self._arxiv_client.__aenter__()
        await self._semantic_scholar_client.__aenter__()
        await self._doi_resolver.__aenter__()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._pubmed_client:
            await self._pubmed_client.__aexit__(exc_type, exc_val, exc_tb)
        if self._arxiv_client:
            await self._arxiv_client.__aexit__(exc_type, exc_val, exc_tb)
        if self._semantic_scholar_client:
            await self._semantic_scholar_client.__aexit__(exc_type, exc_val, exc_tb)
        if self._doi_resolver:
            await self._doi_resolver.__aexit__(exc_type, exc_val, exc_tb)
        if self._pdf_processor:
            await self._pdf_processor.close()

    async def search(
        self,
        keywords: str,
        sources: list[str] | None = None,
        max_results: int = 50,
        min_relevance: float = 0.0,
        extract_full_text: bool = False,
    ) -> list[PaperMetadata]:
        """Search academic databases for papers matching keywords.

        Args:
            keywords: Search keywords or query string
            sources: List of sources to search (default: all available)
            max_results: Maximum number of results per source
            min_relevance: Minimum mental health relevance score (0.0-1.0)
            extract_full_text: Whether to extract full text from PDFs

        Returns:
            List of PaperMetadata objects matching the search criteria
        """
        if sources is None:
            sources = ["pubmed", "arxiv", "semantic_scholar"]

        logger.info(f"Searching for: {keywords} across sources: {sources}")

        all_papers = await self._search_sources(sources, keywords, max_results)

        all_papers = self._filter_and_sort_papers(all_papers, min_relevance)

        if extract_full_text and self.config.extract_full_text:
            await self._enrich_papers_with_full_text(all_papers)

        logger.info(f"Search complete: {len(all_papers)} papers found")
        return all_papers

    async def _search_sources(
        self,
        sources: list[str],
        keywords: str,
        max_results: int,
    ) -> list[PaperMetadata]:
        """Search all requested sources and collect papers."""
        all_papers: list[PaperMetadata] = []

        # Bounded parallelism to avoid overwhelming external services.
        semaphore = asyncio.Semaphore(5)

        async def _safe_search(source: str) -> list[PaperMetadata]:
            try:
                async with semaphore:
                    return await self._search_single_source(source, keywords, max_results)
            except Exception as e:
                logger.error(f"Error searching {source}: {e}")
                return []

        tasks = [_safe_search(source) for source in sources]
        results = await asyncio.gather(*tasks)
        for papers in results:
            all_papers.extend(papers)
        return all_papers

    async def _search_single_source(
        self,
        source: str,
        keywords: str,
        max_results: int,
    ) -> list[PaperMetadata]:
        """Search one source for papers."""
        if source == "pubmed":
            return await self._pubmed_client.search(
                query=keywords,
                max_results=max_results,
            )
        if source == "arxiv":
            return await self._arxiv_client.search(
                query=keywords,
                max_results=max_results,
            )
        if source == "semantic_scholar":
            return await self._semantic_scholar_client.search(
                query=keywords,
                max_results=max_results,
            )
        logger.warning(f"Unknown source: {source}")
        return []

    def _filter_and_sort_papers(
        self,
        papers: list[PaperMetadata],
        min_relevance: float,
    ) -> list[PaperMetadata]:
        """Apply relevance filter and sorting."""
        if self.config.enable_mental_health_filter:
            papers = [paper for paper in papers if paper.mental_health_relevance_score >= min_relevance]
            papers.sort(key=lambda paper: paper.mental_health_relevance_score, reverse=True)
        return papers

    async def _enrich_papers_with_full_text(self, papers: list[PaperMetadata]) -> None:
        """Populate full text and derived fields for papers with PDFs."""
        pdf_papers = [paper for paper in papers if paper.pdf_url]
        if not pdf_papers:
            return

        semaphore = asyncio.Semaphore(max(1, getattr(self.config, "pdf_concurrency", 8)))

        async def _bounded(paper: PaperMetadata) -> None:
            async with semaphore:
                await self._enrich_paper_with_full_text(paper)

        await asyncio.gather(*(_bounded(paper) for paper in pdf_papers))

    async def _enrich_paper_with_full_text(self, paper: PaperMetadata) -> None:
        """Populate full text and extracted insights for a single paper."""
        try:
            full_text = await self._pdf_processor.extract_text(paper.pdf_url)
            if not full_text:
                return
            paper.full_text = full_text
            if self.config.extract_key_findings:
                findings = self._pdf_processor.extract_key_findings(full_text)
                paper.keywords.extend(findings)
            if self.config.extract_methodology and (methodology := self._pdf_processor.extract_methodology(full_text)):
                paper.raw_data["methodology"] = methodology
        except Exception as e:
            logger.warning(f"Failed to extract full text for {paper.paper_id}: {e}")

    async def resolve_doi(self, doi: str) -> PaperMetadata | None:
        """Resolve a DOI and extract metadata.

        Args:
            doi: Digital Object Identifier to resolve

        Returns:
            PaperMetadata object with DOI information
        """
        return await self._doi_resolver.resolve(doi)

    async def get_paper_details(self, paper_id: str, source: str = "semantic_scholar") -> PaperMetadata | None:
        """Get detailed information for a specific paper.

        Args:
            paper_id: Unique identifier for the paper
            source: Source to query (default: semantic_scholar)

        Returns:
            PaperMetadata object with detailed information
        """
        if source == "semantic_scholar":
            # Extract paper ID from format
            if paper_id.startswith("semantic_scholar:"):
                paper_id = paper_id.split(":", 1)[1]

            return await self._semantic_scholar_client.get_paper_details(paper_id)

        logger.warning(f"Paper details not supported for source: {source}")
        return None

    def to_pix32_format(self, papers: list[PaperMetadata]) -> list[dict[str, Any]]:
        """Convert papers to PIX-32 compatible JSON format.

        Args:
            papers: List of PaperMetadata objects

        Returns:
            List of dictionaries in PIX-32 format
        """
        pix32_records = []
        for paper in papers:
            # Convert to Pydantic model for validation
            model = PaperMetadataModel(
                paper_id=paper.paper_id,
                doi=paper.doi,
                pmid=paper.pmid,
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=[AuthorModel(name=a.name, affiliation=a.affiliation) for a in paper.authors],
                publication_date=paper.publication_date.isoformat() if paper.publication_date else None,
                journal=paper.journal,
                volume=paper.volume,
                issue=paper.issue,
                pages=paper.pages,
                abstract=paper.abstract,
                full_text=paper.full_text,
                keywords=paper.keywords,
                study_type=paper.study_type.value,
                access_status=paper.access_status.value,
                pdf_url=paper.pdf_url,
                html_url=paper.html_url,
                citation_count=paper.citation_count,
                references=[
                    CitationModel(paper_id=c.paper_id, title=c.title, year=c.year, venue=c.venue)
                    for c in paper.references
                ],
                cited_by=[
                    CitationModel(paper_id=c.paper_id, title=c.title, year=c.year, venue=c.venue)
                    for c in paper.cited_by
                ],
                mental_health_relevance_score=paper.mental_health_relevance_score,
                mental_health_topics=paper.mental_health_topics,
                source=paper.source.value,
                retrieved_at=paper.retrieved_at.isoformat(),
            )
            pix32_records.append(model.dict())
        return pix32_records

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._pubmed_client.cache.clear()
        self._arxiv_client.cache.clear()
        self._semantic_scholar_client.cache.clear()
        self._doi_resolver.cache.clear()
        logger.info("All caches cleared")


# ============================================================================
# Convenience Functions
# ============================================================================


async def search_papers(
    keywords: str,
    sources: list[str] | None = None,
    max_results: int = 50,
    min_relevance: float = 0.0,
    config: AcademicSourcingConfig | None = None,
) -> list[dict[str, Any]]:
    """Convenience function to search papers and return PIX-32 format.

    Args:
        keywords: Search keywords or query string
        sources: List of sources to search (default: all available)
        max_results: Maximum number of results per source
        min_relevance: Minimum mental health relevance score (0.0-1.0)
        config: Optional configuration object

    Returns:
        List of dictionaries in PIX-32 format
    """
    config = config or AcademicSourcingConfig()
    async with AcademicSourcing(config) as sourcing:
        papers = await sourcing.search(
            keywords=keywords,
            sources=sources,
            max_results=max_results,
            min_relevance=min_relevance,
        )
        return sourcing.to_pix32_format(papers)


async def resolve_doi(doi: str, config: AcademicSourcingConfig | None = None) -> dict[str, Any] | None:
    """Convenience function to resolve a DOI and return PIX-32 format.

    Args:
        doi: Digital Object Identifier to resolve
        config: Optional configuration object

    Returns:
        Dictionary in PIX-32 format or None if resolution fails
    """
    config = config or AcademicSourcingConfig()
    async with AcademicSourcing(config) as sourcing:
        paper = await sourcing.resolve_doi(doi)
        if paper:
            papers = sourcing.to_pix32_format([paper])
            return papers[0] if papers else None
        return None


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main class
    "AcademicSourcing",
    # Configuration
    "AcademicSourcingConfig",
    "AccessStatus",
    # Data classes
    "Author",
    # Pydantic models
    "AuthorModel",
    "CacheConfig",
    "Citation",
    "CitationModel",
    "PaperMetadata",
    "PaperMetadataModel",
    "RateLimitConfig",
    # Enums
    "SourceType",
    "StudyType",
    "resolve_doi",
    # Convenience functions
    "search_papers",
]
