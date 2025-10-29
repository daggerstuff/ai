"""
DOAJ (Directory of Open Access Journals) API client.

Provides search functionality for psychology journals and articles
with therapeutic content.
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from ..models import DatasetSource
from .base_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


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
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize DOAJ client with configuration.
        
        Args:
            enable_cache: Whether to enable request caching
        """
        super().__init__("doaj", enable_cache=enable_cache)
        
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
        subject: str = "Psychology",
        max_results: int = DEFAULT_JOURNAL_RESULTS
    ) -> list[dict[str, Any]]:
        """
        Search for journals by subject.
        
        Args:
            subject: Subject area (e.g., "Psychology", "Psychiatry")
            max_results: Maximum number of journals to return
        
        Returns:
            List of journal dictionaries (empty list on error)
        """
        logger.info(f"Searching DOAJ journals: subject={subject}, max_results={max_results}")
        
        if not subject or not subject.strip():
            logger.warning("Empty subject provided, using default")
            subject = self.DEFAULT_SUBJECT
        
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
        subject: Optional[str] = "Psychology",
        max_results: int = DEFAULT_ARTICLE_RESULTS
    ) -> list[dict[str, Any]]:
        """
        Search for articles by keywords and subject.
        
        Args:
            keywords: Search keywords
            subject: Subject area filter
            max_results: Maximum number of articles to return
        
        Returns:
            List of article dictionaries (empty list on error)
        """
        logger.info(f"Searching DOAJ articles: keywords={keywords}, subject={subject}, max_results={max_results}")
        
        if not keywords:
            logger.warning("No keywords provided for article search")
            return []
        
        max_results = self._validate_max_results(max_results)
        
        # Build keyword query - search in both title and abstract
        keyword_queries = [
            {"match": {field: keyword}}
            for keyword in keywords
            for field in ["bibjson.title", "bibjson.abstract"]
        ]
        
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
    
    def _extract_article_metadata(self, article: dict[str, Any]) -> Optional[dict[str, Any]]:
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
            pub_date_str = bibjson.get("year", "")
            try:
                pub_date = datetime(int(pub_date_str), 1, 1) if pub_date_str else datetime.now()
            except (ValueError, TypeError):
                pub_date = datetime.now()
            
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
            
            return {
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "publication_date": pub_date,
                "doi": doi,
                "url": url or f"https://doaj.org/article/{article.get('id', '')}",
                "keywords": keywords,
                "journal_title": journal_title,
                "doaj_id": article.get("id", "")
            }
        
        except Exception as e:
            logger.warning(f"Failed to extract article metadata: {e}")
            return None
    
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
        
        # Generate source ID
        doaj_id = metadata.get("doaj_id", "")
        source_id = f"doaj_{hashlib.md5(doaj_id.encode()).hexdigest()[:12]}"
        
        return DatasetSource(
            source_id=source_id,
            title=metadata["title"],
            authors=metadata["authors"],
            publication_date=metadata["publication_date"],
            source_type="journal",
            url=metadata["url"],
            doi=metadata.get("doi"),
            abstract=metadata["abstract"],
            keywords=metadata["keywords"],
            open_access=True,  # All DOAJ articles are open access
            data_availability="unknown",  # Needs manual verification
            discovery_date=datetime.now(),
            discovery_method="doaj_manual"
        )
    
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
        """
        logger.info(f"Searching therapeutic content: keywords={keywords}, max_results={max_results}")
        
        # Search articles
        articles = self.search_articles(
            keywords=keywords,
            subject="Psychology",
            max_results=max_results
        )
        
        # Convert to DatasetSource objects
        sources = []
        for article in articles:
            source = self.convert_to_dataset_source(article)
            if source:
                sources.append(source)
        
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
        """
        logger.info(
            f"Investigating journal {journal_issn} for therapeutic content: "
            f"keywords={therapeutic_keywords}, max_articles={max_articles}"
        )
        
        # Get all articles from journal
        articles = self.get_journal_articles(journal_issn, max_articles)
        
        # Filter for therapeutic content
        therapeutic_sources = []
        for article in articles:
            metadata = self._extract_article_metadata(article)
            if not metadata:
                continue
            
            # Check if article contains therapeutic keywords
            text_to_search = " ".join([
                metadata["title"],
                metadata["abstract"],
                *metadata["keywords"]
            ]).lower()
            
            has_therapeutic_content = any(
                keyword.lower() in text_to_search
                for keyword in therapeutic_keywords
            )
            
            if has_therapeutic_content:
                source = self.convert_to_dataset_source(article)
                if source:
                    therapeutic_sources.append(source)
        
        logger.info(
            f"Found {len(therapeutic_sources)} articles with therapeutic content "
            f"in journal {journal_issn}"
        )
        return therapeutic_sources
