"""
Base Publisher Class for Academic Sourcing Engine

This module defines the abstract base class for all academic publisher integrations,
providing common functionality for searching, retrieving, and processing academic books.
"""

import abc
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BookFormat(Enum):
    """Supported book formats for academic sourcing"""
    PDF = "pdf"
    EPUB = "epub"
    HTML = "html"
    XML = "xml"
    PLAIN_TEXT = "txt"
    JSON = "json"

@dataclass
class BookMetadata:
    """Metadata for an academic book"""
    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    isbn: Optional[str] = None
    doi: Optional[str] = None
    abstract: Optional[str] = None
    keywords: Optional[List[str]] = None
    chapters: Optional[List[Dict[str, Any]]] = None
    page_count: Optional[int] = None
    language: Optional[str] = "en"
    license: Optional[str] = None
    copyright_status: Optional[str] = None
    therapeutic_relevance_score: Optional[float] = None
    stage_assignment: Optional[str] = None  # stage1_foundation, stage2_therapeutic_expertise, etc.
    quality_score: Optional[float] = None
    safety_score: Optional[float] = None
    bias_score: Optional[float] = None
    source_publisher: Optional[str] = None
    raw_metadata: Optional[Dict[str, Any]] = None

@dataclass
class BookContent:
    """Content of an academic book with metadata"""
    metadata: BookMetadata
    content: str
    format: BookFormat
    chapter_contents: Optional[List[Tuple[str, str]]] = None  # List of (chapter_title, chapter_content)
    therapeutic_concepts: Optional[List[str]] = None
    anonymized_content: Optional[str] = None

class BasePublisher(abc.ABC):
    """Abstract base class for academic publisher integrations"""

    def __init__(self, name: str, api_base_url: str, requires_auth: bool = False):
        """
        Initialize a publisher integration

        Args:
            name: Name of the publisher
            api_base_url: Base URL for the publisher's API
            requires_auth: Whether the publisher requires authentication
        """
        self.name = name
        self.api_base_url = api_base_url
        self.requires_auth = requires_auth
        self.session = self._create_session()
        self._auth_token = None

        logger.info(f"Initialized {self.name} publisher integration")

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    def _get_headers(self) -> Dict[str, str]:
        """Get default headers for API requests"""
        headers = {
            "User-Agent": f"PixelatedEmpathyAcademicSourcing/1.0 ({self.name} Integration)",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        return headers

    def authenticate(self, api_key: Optional[str] = None, **kwargs) -> bool:
        """
        Authenticate with the publisher's API

        Args:
            api_key: API key for authentication
            **kwargs: Additional authentication parameters

        Returns:
            bool: True if authentication succeeded, False otherwise
        """
        if not self.requires_auth:
            logger.info(f"{self.name} does not require authentication")
            return True

        try:
            auth_result = self._authenticate(api_key, **kwargs)
            if auth_result:
                logger.info(f"Successfully authenticated with {self.name}")
            else:
                logger.warning(f"Failed to authenticate with {self.name}")
            return auth_result
        except Exception as e:
            logger.error(f"Authentication error with {self.name}: {e}")
            return False

    @abc.abstractmethod
    def _authenticate(self, api_key: Optional[str] = None, **kwargs) -> bool:
        """Publisher-specific authentication implementation"""
        pass

    @abc.abstractmethod
    def search_books(
        self,
        query: str,
        year_range: Optional[Tuple[int, int]] = None,
        therapeutic_topics: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[BookMetadata]:
        """
        Search for books in the publisher's catalog

        Args:
            query: Search query (title, author, keywords, etc.)
            year_range: Tuple of (start_year, end_year) to filter by publication year
            therapeutic_topics: List of therapeutic topics to filter by
            limit: Maximum number of results to return
            offset: Offset for pagination

        Returns:
            List of BookMetadata objects matching the search criteria
        """
        pass

    @abc.abstractmethod
    def get_book_metadata(self, identifier: str) -> Optional[BookMetadata]:
        """
        Get metadata for a specific book

        Args:
            identifier: Book identifier (ISBN, DOI, or publisher-specific ID)

        Returns:
            BookMetadata object if found, None otherwise
        """
        pass

    @abc.abstractmethod
    def get_book_content(self, identifier: str, format: BookFormat = BookFormat.PLAIN_TEXT) -> Optional[BookContent]:
        """
        Get content for a specific book

        Args:
            identifier: Book identifier (ISBN, DOI, or publisher-specific ID)
            format: Desired format for the book content

        Returns:
            BookContent object if found and accessible, None otherwise
        """
        pass

    @abc.abstractmethod
    def get_chapter_content(self, identifier: str, chapter_id: str) -> Optional[str]:
        """
        Get content for a specific chapter of a book

        Args:
            identifier: Book identifier (ISBN, DOI, or publisher-specific ID)
            chapter_id: Chapter identifier

        Returns:
            Chapter content as string if found, None otherwise
        """
        pass

    def assess_therapeutic_relevance(self, metadata: BookMetadata) -> float:
        """
        Assess the therapeutic relevance of a book based on its metadata

        Args:
            metadata: BookMetadata object to assess

        Returns:
            float: Relevance score between 0.0 and 1.0
        """
        score = 0.0

        # Check for therapeutic keywords in title, abstract, and keywords
        therapeutic_keywords = [
            "psychology", "psychotherapy", "therapy", "therapeutic", "mental health",
            "counseling", "clinical psychology", "psychiatry", "trauma", "anxiety",
            "depression", "cognitive behavioral therapy", "cbt", "dialectical behavior therapy",
            "dbt", "mindfulness", "emotional regulation", "personality disorders",
            "crisis intervention", "suicide prevention", "addiction", "substance abuse",
            "eating disorders", "ptsd", "post-traumatic stress", "neuropsychology",
            "child psychology", "adolescent psychology", "family therapy", "group therapy",
            "evidence-based practice", "clinical techniques", "diagnostic", "dsm-5",
            "assessment", "intervention", "treatment planning", "ethics", "cultural competence"
        ]

        # Check title
        title = metadata.title.lower()
        title_matches = sum(1 for keyword in therapeutic_keywords if keyword in title)
        if title_matches > 0:
            score += 0.3

        # Check abstract
        if metadata.abstract:
            abstract = metadata.abstract.lower()
            abstract_matches = sum(1 for keyword in therapeutic_keywords if keyword in abstract)
            if abstract_matches > 0:
                score += 0.4

        # Check keywords
        if metadata.keywords:
            keyword_matches = sum(1 for keyword in metadata.keywords if keyword.lower() in therapeutic_keywords)
            if keyword_matches > 0:
                score += 0.3

        # Normalize score to 0-1 range
        score = min(score, 1.0)

        # Assign stage based on relevance score
        if score >= 0.8:
            metadata.stage_assignment = "stage2_therapeutic_expertise"
        elif score >= 0.5:
            metadata.stage_assignment = "stage1_foundation"
        else:
            metadata.stage_assignment = "stage1_foundation"  # Default to foundation

        metadata.therapeutic_relevance_score = score
        return score

    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Make an API request to the publisher's API

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: Query parameters
            data: Form data
            json: JSON data
            timeout: Request timeout in seconds

        Returns:
            Response JSON as dict if successful, None otherwise
        """
        url = f"{self.api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"

        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self._get_headers(),
                params=params,
                data=data,
                json=json,
                timeout=timeout
            )

            response.raise_for_status()

            if response.status_code == 204:  # No content
                return {}

            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {self.name} API failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response status: {e.response.status_code}")
                logger.error(f"Response content: {e.response.text}")
            return None

    def __str__(self) -> str:
        return f"{self.name} Publisher Integration"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', api_base_url='{self.api_base_url}')>"