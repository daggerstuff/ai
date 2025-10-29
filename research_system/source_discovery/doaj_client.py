"""
DOAJ (Directory of Open Access Journals) API client.

Provides search functionality for psychology journals and articles
with therapeutic content.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, TypedDict, NotRequired

from ..models import DatasetSource
from .base_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class ArticleMetadata(TypedDict):
    """Structured type for article metadata."""
    title: str
    abstract: str
    authors: list[str]
    publication_date: Optional[datetime]
    doi: NotRequired[str]
    url: str
    keywords: list[str]
    journal_title: str
    doaj_id: str


class MetadataExtractor(ABC):
    """Base class for metadata extraction strategies."""
    
    @abstractmethod
    def extract(self, article: dict[str, Any]) -> Optional[ArticleMetadata]:
        """Extract metadata from article."""
        pass


class DOAJMetadataExtractor(MetadataExtractor):
    """DOAJ-specific metadata extraction."""
    
    DEFAULT_MONTH = 1
    DEFAULT_DAY = 1
    
    def extract(self, article: dict[str, Any]) -> Optional[ArticleMetadata]:
        """
        Extract relevant metadata from DOAJ article.
        
        Args:
            article: DOAJ article dictionary
        
        Returns:
            Extracted metadata dictionary or None
        """
        try:
            bibjson = article.get("bibjson", {})
            
            # Extract basic info
            title = bibjson.get("title", "")
            abstract = bibjson.get("abstract", "")
            
            # Extract authors
            authors = []
            for author in bibjson.get("author", []):
                name = author.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract publication date
            pub_date = None
            pub_date_str = bibjson.get("year", "")
            if pub_date_str:
                try:
                    pub_date = datetime(int(pub_date_str), self.DEFAULT_MONTH, self.DEFAULT_DAY)
                except (ValueError, TypeError):
                    logger.debug(f"Invalid publication date '{pub_date_str}', using None")
            
            # Extract identifiers
            doi = None
            url = None
            for identifier in bibjson.get("identifier", []):
                if identifier.get("type") == "doi":
                    doi = identifier.get("id")
                elif identifier.get("type") == "url":
                    url = identifier.get("id")
            
            # Extract keywords/subjects
            keywords = []
            for subject in bibjson.get("subject", []):
                term = subject.get("term", "")
                if term:
                    keywords.append(term)
            
            # Extract journal info
            journal = bibjson.get("journal", {})
            journal_title = journal.get("title", "")
            
            metadata: ArticleMetadata = {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": pub_date,
                "url": url or f"https://doaj.org/article/{article.get('id', '')}",
                "keywords": keywords,
                "journal_title": journal_title,
                "doaj_id": article.get("id", "")
            }
            
            if doi:
                metadata["doi"] = doi
            
            return metadata
        
        except (KeyError, TypeError, AttributeError, ValueError) as e:
            logger.warning(f"Failed to extract article metadata: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error extracting article metadata: {e}", exc_info=True)
            return None


class DOAJClient(BaseAPIClient):
    """Client for searching DOAJ for open access psychology journals."""
    
    # API endpoints
    ENDPOINT_JOURNALS = "search/journals"
    ENDPOINT_ARTICLES = "search/articles"
    
    # Default values
    DEFAULT_SUBJECT = "Psychology"
    DEFAULT_JOURNAL_RESULTS = 50
    DEFAULT_ARTICLE_RESULTS = 100
    
    # Validation limits
    MIN_RESULTS = 1
    MAX_RESULTS_LIMIT = 1000
    
    # Field names
    FIELD_SUBJECT_TERM = "bibjson.subject.term.exact"
    FIELD_JOURNAL_ISSN = "bibjson.journal.issn.exact"
    
    # Discovery methods
    DISCOVERY_METHOD_MANUAL = "doaj_manual"
    DISCOVERY_METHOD_AUTOMATED = "doaj_automated"
    DEFAULT_DISCOVERY_METHOD = DISCOVERY_METHOD_AUTOMATED
    
    # Required metadata fields for DatasetSource creation
    REQUIRED_METADATA_FIELDS = frozenset([
        'title', 'authors', 'publication_date', 
        'url', 'abstract', 'keywords'
    ])
    
    # Search field names
    SEARCH_FIELDS = ["bibjson.title", "bibjson.abstract"]
    
    def __init__(self, enable_cache: bool = True, extractor: Optional[MetadataExtractor] = None):
        """
        Initialize DOAJ client with configuration.
        
        Args:
            enable_cache: Whether to enable request caching
            extractor: Custom metadata extractor (defaults to DOAJMetadataExtractor)
        """
        super().__init__("doaj", enable_cache=enable_cache)
        self._extractor = extractor or DOAJMetadataExtractor()
        
        # Set API key header if available
        api_key = self.config.get("api_endpoints.doaj.api_key", "")
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def _validate_max_results(self, max_results: int) -> int:
        """
        Validate and normalize max_results parameter.
        
        Args:
            max_results: Requested maximum results
        
        Returns:
            Validated max_results value (clamped to valid range)
        """
        if max_results < self.MIN_RESULTS or max_results > self.MAX_RESULTS_LIMIT:
            logger.warning(
                f"max_results {max_results} out of range [{self.MIN_RESULTS}, {self.MAX_RESULTS_LIMIT}], "
                f"capping to valid range"
            )
            return max(self.MIN_RESULTS, min(max_results, self.MAX_RESULTS_LIMIT))
        return max_results
    
    def _validate_subject(self, subject: Optional[str]) -> str:
        """
        Validate and normalize subject parameter.
        
        Args:
            subject: Subject to validate
        
        Returns:
            Validated subject string (defaults to DEFAULT_SUBJECT if invalid)
        """
        if subject is None:
            return self.DEFAULT_SUBJECT
        
        if not isinstance(subject, str):
            logger.warning(f"Invalid subject type: {type(subject)}, converting to string")
            subject = str(subject)
        
        if not subject.strip():
            logger.warning("Empty subject provided, using default")
            return self.DEFAULT_SUBJECT
        
        return subject.strip()
    
    def _validate_keywords(self, keywords: Any, param_name: str = "keywords") -> list[str]:
        """
        Validate and clean keyword list.
        
        Args:
            keywords: Keywords to validate (should be list of strings)
            param_name: Parameter name for logging
        
        Returns:
            Cleaned list of non-empty string keywords
        """
        if not isinstance(keywords, list):
            logger.warning(f"Invalid {param_name} type: {type(keywords)}, expected list")
            return []
        
        cleaned = [k.strip() for k in keywords if isinstance(k, str) and k.strip()]
        
        if not cleaned:
            logger.warning(f"No valid {param_name} provided")
        
        return cleaned
    
    def _build_term_query(
        self,
        field: str,
        value: str,
        size: int,
        sort_field: str = "created_date"
    ) -> dict[str, Any]:
        """
        Build standard term query for DOAJ API.
        
        Args:
            field: Field name to query
            value: Value to match
            size: Number of results
            sort_field: Field to sort by
        
        Returns:
            Query dictionary
        """
        return {
            "query": {
                "bool": {
                    "must": [{"term": {field: value}}]
                }
            },
            "size": size,
            "sort": [{sort_field: {"order": "desc"}}]
        }
    
    def _build_keyword_queries(
        self, 
        keywords: list[str], 
        search_fields: Optional[list[str]] = None
    ) -> list[dict[str, Any]]:
        """
        Build keyword match queries for multiple fields.
        
        Args:
            keywords: Search keywords
            search_fields: Fields to search in (defaults to SEARCH_FIELDS)
        
        Returns:
            List of match query dictionaries
        """
        if search_fields is None:
            search_fields = self.SEARCH_FIELDS
        
        return [
            {"match": {field: keyword}}
            for keyword in keywords
            for field in search_fields
        ]
    
    def _execute_search(
        self,
        endpoint: str,
        query: dict[str, Any],
        operation_name: str,
        raise_on_error: bool = False
    ) -> list[dict[str, Any]]:
        """
        Execute search query with standardized error handling.
        
        Args:
            endpoint: API endpoint
            query: Search query dictionary
            operation_name: Name for logging
            raise_on_error: If True, raise exceptions instead of returning empty list
        
        Returns:
            List of results or empty list on error (if raise_on_error=False)
        
        Raises:
            APIError: If raise_on_error=True and request fails
        """
        logger.info(f"Executing {operation_name}")
        
        try:
            response = self._make_request(endpoint, params=query, method="POST")
            data = response.json()
            results = data.get("results", [])
            logger.info(f"{operation_name} returned {len(results)} results")
            return results
        
        except APIError as e:
            logger.error(f"{operation_name} failed: {e}")
            if raise_on_error:
                raise
            return []
        except (ValueError, KeyError, TypeError) as e:
            logger.error(f"Data parsing error in {operation_name}: {e}", exc_info=True)
            if raise_on_error:
                raise APIError(f"Data parsing failed: {e}") from e
            return []
    
    def search_journals(
        self,
        subject: Optional[str] = None,
        max_results: int = DEFAULT_JOURNAL_RESULTS
    ) -> list[dict[str, Any]]:
        """
        Search for journals by subject.
        
        Args:
            subject: Subject area (defaults to DEFAULT_SUBJECT)
            max_results: Maximum number of journals to return
        
        Returns:
            List of journal dictionaries (empty list on error)
        """
        subject = self._validate_subject(subject)
        logger.info(f"Searching DOAJ journals: subject={subject}, max_results={max_results}")
        
        max_results = self._validate_max_results(max_results)
        
        query = self._build_term_query(
            self.FIELD_SUBJECT_TERM,
            subject,
            max_results
        )
        
        return self._execute_search(
            self.ENDPOINT_JOURNALS,
            query,
            f"DOAJ {subject} journal search"
        )
    
    def search_articles(
        self,
        keywords: list[str],
        subject: Optional[str] = None,
        max_results: int = DEFAULT_ARTICLE_RESULTS
    ) -> list[dict[str, Any]]:
        """
        Search for articles by keywords and subject.
        
        Args:
            keywords: Search keywords
            subject: Subject area filter (defaults to DEFAULT_SUBJECT)
            max_results: Maximum number of articles to return
        
        Returns:
            List of article dictionaries (empty list on error)
        """
        subject = self._validate_subject(subject)
        
        # Validate and clean keywords
        keywords = self._validate_keywords(keywords, "keywords")
        if not keywords:
            return []
        
        logger.info(f"Searching DOAJ articles: keywords={keywords}, subject={subject}, max_results={max_results}")
        
        max_results = self._validate_max_results(max_results)
        
        # Build keyword query - search in both title and abstract
        keyword_queries = self._build_keyword_queries(keywords)
        
        # Build full query
        query_parts = {
            "query": {
                "bool": {
                    "should": keyword_queries,
                    "minimum_should_match": 1
                }
            },
            "size": max_results,
            "sort": [{"created_date": {"order": "desc"}}]
        }
        
        # Add subject filter if specified
        if subject:
            query_parts["query"]["bool"]["must"] = [
                {"term": {self.FIELD_SUBJECT_TERM: subject}}
            ]
        
        return self._execute_search(
            self.ENDPOINT_ARTICLES,
            query_parts,
            f"DOAJ article search with keywords: {keywords}"
        )
    
    def get_journal_articles(
        self,
        journal_issn: str,
        max_results: int = DEFAULT_ARTICLE_RESULTS
    ) -> list[dict[str, Any]]:
        """
        Get articles from a specific journal by ISSN.
        
        Args:
            journal_issn: Journal ISSN
            max_results: Maximum number of articles to return
        
        Returns:
            List of article dictionaries (empty list on error)
        """
        logger.info(f"Fetching DOAJ articles: journal_issn={journal_issn}, max_results={max_results}")
        
        if not journal_issn or not journal_issn.strip():
            logger.warning("Empty journal ISSN provided")
            return []
        
        max_results = self._validate_max_results(max_results)
        
        query = self._build_term_query(
            self.FIELD_JOURNAL_ISSN,
            journal_issn,
            max_results
        )
        
        return self._execute_search(
            self.ENDPOINT_ARTICLES,
            query,
            f"DOAJ articles from journal {journal_issn}"
        )
    
    def _extract_article_metadata(self, article: dict[str, Any]) -> Optional[ArticleMetadata]:
        """
        Extract relevant metadata from DOAJ article using configured extractor.
        
        Args:
            article: DOAJ article dictionary
        
        Returns:
            Extracted metadata dictionary or None
        """
        return self._extractor.extract(article)
    
    def _create_source_from_metadata(
        self, 
        article: dict[str, Any], 
        metadata: ArticleMetadata,
        discovery_method: str = DEFAULT_DISCOVERY_METHOD,
        skip_validation: bool = False
    ) -> Optional[DatasetSource]:
        """
        Create DatasetSource from pre-extracted metadata.
        
        Args:
            article: Original DOAJ article dictionary
            metadata: Pre-extracted metadata dictionary
            discovery_method: Discovery method override (defaults to DEFAULT_DISCOVERY_METHOD)
            skip_validation: Skip field validation if metadata already validated
        
        Returns:
            DatasetSource object or None if creation fails
        """
        try:
            # Validate required fields unless skipped
            if not skip_validation:
                missing_fields = [
                    field for field in self.REQUIRED_METADATA_FIELDS 
                    if field not in metadata
                ]
                
                if missing_fields:
                    logger.warning(f"Missing required metadata fields: {missing_fields}")
                    return None
            
            # Use current date if publication date is missing
            pub_date = metadata.get("publication_date")
            if not pub_date:
                pub_date = datetime.now()
            
            # Generate source ID
            doaj_id = metadata.get("doaj_id", "")
            if not doaj_id:
                logger.warning("Missing DOAJ ID for article")
                return None
            
            source_id = f"doaj_{hashlib.sha256(doaj_id.encode()).hexdigest()[:12]}"
            
            return DatasetSource(
                source_id=source_id,
                title=metadata["title"],
                authors=metadata["authors"],
                publication_date=pub_date,
                source_type="journal",
                url=metadata["url"],
                doi=metadata.get("doi"),
                abstract=metadata["abstract"],
                keywords=metadata["keywords"],
                open_access=True,
                data_availability="unknown",
                discovery_date=datetime.now(),
                discovery_method=discovery_method
            )
        except (KeyError, TypeError) as e:
            logger.error(f"Failed to create DatasetSource from metadata: {e}", exc_info=True)
            return None
    
    def _convert_articles_to_sources(
        self,
        articles: list[dict[str, Any]],
        filter_fn: Optional[Callable[[dict[str, Any]], bool]] = None
    ) -> list[DatasetSource]:
        """
        Convert articles to DatasetSource objects with optional filtering.
        
        Args:
            articles: List of DOAJ article dictionaries
            filter_fn: Optional function to filter articles (receives metadata dict)
        
        Returns:
            List of DatasetSource objects
        """
        sources = []
        for article in articles:
            metadata = self._extract_article_metadata(article)
            if not metadata:
                continue
            
            # Apply filter if provided
            if filter_fn and not filter_fn(metadata):
                continue
            
            # Convert using pre-extracted metadata to avoid duplicate extraction
            source = self._create_source_from_metadata(article, metadata)
            if source:
                sources.append(source)
        
        return sources
    
    def convert_to_dataset_source(self, article: dict[str, Any]) -> Optional[DatasetSource]:
        """
        Convert DOAJ article to DatasetSource object.
        
        Args:
            article: DOAJ article dictionary
        
        Returns:
            DatasetSource object or None if conversion fails
        """
        metadata = self._extract_article_metadata(article)
        if not metadata:
            return None
        
        return self._create_source_from_metadata(article, metadata)
    
    def search_therapeutic_content(
        self,
        keywords: list[str],
        max_results: int = DEFAULT_ARTICLE_RESULTS
    ) -> list[DatasetSource]:
        """
        Search for therapeutic content and convert to DatasetSource objects.
        
        Args:
            keywords: Search keywords for therapeutic content
            max_results: Maximum number of results
        
        Returns:
            List of DatasetSource objects
        
        Raises:
            APIError: If API request fails with raise_on_error=True
        """
        # Validate keywords
        keywords = self._validate_keywords(keywords, "keywords")
        if not keywords:
            return []
        
        logger.info(f"Searching therapeutic content: keywords={keywords}, max_results={max_results}")
        
        # Search articles
        articles = self.search_articles(
            keywords=keywords,
            subject=self.DEFAULT_SUBJECT,
            max_results=max_results
        )
        
        # Convert to DatasetSource objects
        sources = self._convert_articles_to_sources(articles)
        
        logger.info(f"Converted {len(sources)} DOAJ articles to DatasetSource objects")
        return sources
    
    def investigate_journal(
        self,
        journal_issn: str,
        therapeutic_keywords: list[str],
        max_articles: int = DEFAULT_ARTICLE_RESULTS
    ) -> list[DatasetSource]:
        """
        Investigate a specific journal for therapeutic content.
        
        Args:
            journal_issn: Journal ISSN to investigate
            therapeutic_keywords: Keywords to filter therapeutic content
            max_articles: Maximum articles to retrieve
        
        Returns:
            List of DatasetSource objects with therapeutic content
        
        Raises:
            APIError: If API request fails with raise_on_error=True
        """
        # Validate keywords
        therapeutic_keywords = self._validate_keywords(therapeutic_keywords, "therapeutic_keywords")
        if not therapeutic_keywords:
            return []
        
        logger.info(
            f"Investigating journal {journal_issn} for therapeutic content: "
            f"keywords={therapeutic_keywords}, max_articles={max_articles}"
        )
        
        # Get all articles from journal
        articles = self.get_journal_articles(journal_issn, max_articles)
        
        # Pre-compute lowercase keywords once for all articles (performance optimization)
        therapeutic_keywords_lower = [k.lower() for k in therapeutic_keywords]
        
        # Define filter function for therapeutic content
        def has_therapeutic_content(metadata: ArticleMetadata) -> bool:
            """Check if article metadata contains therapeutic keywords."""
            # Check each field separately to short-circuit on first match
            title = metadata.get('title', '').lower()
            if any(keyword in title for keyword in therapeutic_keywords_lower):
                return True
            
            abstract = metadata.get('abstract', '').lower()
            if any(keyword in abstract for keyword in therapeutic_keywords_lower):
                return True
            
            keywords_text = ' '.join(metadata.get('keywords', [])).lower()
            return any(keyword in keywords_text for keyword in therapeutic_keywords_lower)
        
        # Convert and filter articles
        therapeutic_sources = self._convert_articles_to_sources(articles, has_therapeutic_content)
        
        logger.info(
            f"Found {len(therapeutic_sources)} articles with therapeutic content "
            f"in journal {journal_issn}"
        )
        return therapeutic_sources
