"""
Metadata parser for extracting and normalizing dataset metadata from various sources.

Handles different metadata formats and provides unified parsing interface.
"""

import logging
import re
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MetadataParser:
    """Parser for extracting and normalizing dataset metadata."""
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize title for comparison.
        
        Args:
            title: Raw title string
        
        Returns:
            Normalized title
        """
        # Convert to lowercase
        normalized = title.lower()
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Remove punctuation at start/end
        normalized = normalized.strip(".,;:!?-")
        
        return normalized
    
    @staticmethod
    def normalize_doi(doi: Optional[str]) -> Optional[str]:
        """
        Normalize DOI for comparison.
        
        Args:
            doi: Raw DOI string
        
        Returns:
            Normalized DOI or None
        """
        if not doi:
            return None
        
        # Remove common prefixes
        doi = doi.lower()
        doi = re.sub(r'^(doi:|https?://doi\.org/|https?://dx\.doi\.org/)', '', doi)
        
        # Remove whitespace
        doi = doi.strip()
        
        return doi if doi else None
    
    @staticmethod
    def normalize_author_name(name: str) -> str:
        """
        Normalize author name for comparison.
        
        Args:
            name: Raw author name
        
        Returns:
            Normalized author name
        """
        # Convert to lowercase
        normalized = name.lower()
        
        # Remove titles and suffixes
        titles = ['dr', 'prof', 'mr', 'mrs', 'ms', 'phd', 'md', 'jr', 'sr']
        for title in titles:
            normalized = re.sub(rf'\b{title}\.?\b', '', normalized)
        
        # Remove extra whitespace
        normalized = " ".join(normalized.split())
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        return normalized.strip()
    
    @staticmethod
    def extract_author_last_names(authors: list[str]) -> set[str]:
        """
        Extract last names from author list.
        
        Args:
            authors: List of author names
        
        Returns:
            Set of normalized last names
        """
        last_names = set()
        
        for author in authors:
            # Normalize name
            normalized = MetadataParser.normalize_author_name(author)
            
            # Split into parts
            parts = normalized.split()
            
            if parts:
                # Assume last part is last name
                last_names.add(parts[-1])
        
        return last_names
    
    @staticmethod
    def parse_date(date_str: str) -> Optional[datetime]:
        """
        Parse date string in various formats.
        
        Args:
            date_str: Date string
        
        Returns:
            Datetime object or None
        """
        if not date_str:
            return None
        
        # Try common date formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y/%m/%d",
            "%d/%m/%Y",
            "%m/%d/%Y",
            "%Y",
            "%B %d, %Y",
            "%b %d, %Y",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Try parsing year only
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        if year_match:
            try:
                year = int(year_match.group(0))
                return datetime(year, 1, 1)
            except ValueError:
                pass
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    @staticmethod
    def extract_keywords(text: str, min_length: int = 3) -> list[str]:
        """
        Extract potential keywords from text.
        
        Args:
            text: Text to extract keywords from
            min_length: Minimum keyword length
        
        Returns:
            List of keywords
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except hyphens
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Split into words
        words = text.split()
        
        # Filter by length and remove common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'they', 'their', 'them', 'we', 'our', 'us'
        }
        
        keywords = [
            word for word in words
            if len(word) >= min_length and word not in stop_words
        ]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for keyword in keywords:
            if keyword not in seen:
                seen.add(keyword)
                unique_keywords.append(keyword)
        
        return unique_keywords
    
    @staticmethod
    def extract_urls(text: str) -> list[str]:
        """
        Extract URLs from text.
        
        Args:
            text: Text to extract URLs from
        
        Returns:
            List of URLs
        """
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, text)
        return urls
    
    @staticmethod
    def extract_dois(text: str) -> list[str]:
        """
        Extract DOIs from text.
        
        Args:
            text: Text to extract DOIs from
        
        Returns:
            List of normalized DOIs
        """
        # DOI pattern: 10.xxxx/xxxxx
        doi_pattern = r'\b10\.\d{4,}/[^\s<>"{}|\\^`\[\]]+'
        dois = re.findall(doi_pattern, text)
        
        # Normalize DOIs
        normalized_dois = [MetadataParser.normalize_doi(doi) for doi in dois]
        return [doi for doi in normalized_dois if doi]
    
    @staticmethod
    def clean_abstract(abstract: str, max_length: int = 5000) -> str:
        """
        Clean and truncate abstract text.
        
        Args:
            abstract: Raw abstract text
            max_length: Maximum length
        
        Returns:
            Cleaned abstract
        """
        # Remove HTML tags
        abstract = re.sub(r'<[^>]+>', '', abstract)
        
        # Remove extra whitespace
        abstract = " ".join(abstract.split())
        
        # Truncate if too long
        if len(abstract) > max_length:
            abstract = abstract[:max_length] + "..."
        
        return abstract
    
    @staticmethod
    def parse_license(license_text: str) -> dict[str, Any]:
        """
        Parse license information.
        
        Args:
            license_text: License text or identifier
        
        Returns:
            Dictionary with license information
        """
        license_text_lower = license_text.lower()
        
        # Common open licenses
        open_licenses = {
            'cc0': {'name': 'CC0 1.0', 'commercial': True, 'ai_training': True},
            'cc-by': {'name': 'CC BY', 'commercial': True, 'ai_training': True},
            'cc-by-sa': {'name': 'CC BY-SA', 'commercial': True, 'ai_training': True},
            'cc-by-nc': {'name': 'CC BY-NC', 'commercial': False, 'ai_training': True},
            'mit': {'name': 'MIT License', 'commercial': True, 'ai_training': True},
            'apache': {'name': 'Apache License', 'commercial': True, 'ai_training': True},
            'bsd': {'name': 'BSD License', 'commercial': True, 'ai_training': True},
            'gpl': {'name': 'GPL', 'commercial': True, 'ai_training': True},
            'public domain': {'name': 'Public Domain', 'commercial': True, 'ai_training': True},
        }
        
        for key, info in open_licenses.items():
            if key in license_text_lower:
                return {
                    'license': info['name'],
                    'allows_commercial': info['commercial'],
                    'allows_ai_training': info['ai_training'],
                    'raw_text': license_text
                }
        
        # Unknown license - assume restrictive
        return {
            'license': license_text,
            'allows_commercial': False,
            'allows_ai_training': False,
            'raw_text': license_text
        }
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using Jaccard similarity.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score (0-1)
        """
        # Normalize texts
        text1 = text1.lower()
        text2 = text2.lower()
        
        # Extract words
        words1 = set(re.findall(r'\w+', text1))
        words2 = set(re.findall(r'\w+', text2))
        
        # Calculate Jaccard similarity
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def extract_data_availability_info(text: str) -> dict[str, Any]:
        """
        Extract data availability information from text.
        
        Args:
            text: Text containing data availability information
        
        Returns:
            Dictionary with availability information
        """
        text_lower = text.lower()
        
        # Check for availability indicators
        available_indicators = [
            'data available',
            'data are available',
            'data is available',
            'openly available',
            'publicly available',
            'available upon request',
            'available on request',
            'data sharing',
            'supplementary data',
            'supporting data'
        ]
        
        restricted_indicators = [
            'data not available',
            'data unavailable',
            'restricted access',
            'confidential',
            'proprietary',
            'privacy concerns',
            'ethical restrictions'
        ]
        
        has_available = any(indicator in text_lower for indicator in available_indicators)
        has_restricted = any(indicator in text_lower for indicator in restricted_indicators)
        
        if has_restricted:
            status = 'restricted'
        elif has_available:
            if 'upon request' in text_lower or 'on request' in text_lower:
                status = 'upon_request'
            else:
                status = 'available'
        else:
            status = 'unknown'
        
        # Extract URLs that might point to data
        urls = MetadataParser.extract_urls(text)
        data_urls = [
            url for url in urls
            if any(domain in url.lower() for domain in [
                'github.com', 'figshare.com', 'zenodo.org', 'dryad',
                'osf.io', 'dataverse', 'data.', '/data/', '/dataset/'
            ])
        ]
        
        return {
            'status': status,
            'data_urls': data_urls,
            'raw_text': text
        }
