"""
PubMed Central API client using NCBI E-utilities.

Provides search functionality for mental health and therapy-related datasets
with support for MeSH terms, open access filtering, and pagination.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, Optional

from ..models import DatasetSource
from .base_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class PubMedClient(BaseAPIClient):
    """Client for searching PubMed Central using NCBI E-utilities API."""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize PubMed client with configuration.
        
        Args:
            enable_cache: Whether to enable request caching
        """
        super().__init__("pubmed", enable_cache=enable_cache)
        
        self.api_key = self.config.get("api_endpoints.pubmed.api_key", "")
        self.email = self.config.get("api_endpoints.pubmed.email", "")
        
        if not self.email:
            logger.warning("NCBI_EMAIL not configured. This is required for API usage.")
    
    def _make_pubmed_request(
        self,
        endpoint: str,
        params: dict[str, str],
        use_cache: bool = True
    ) -> Any:
        """
        Make PubMed-specific request with API key and email.
        
        Args:
            endpoint: E-utilities endpoint (esearch, efetch, etc.)
            params: Query parameters
            use_cache: Whether to use cache
        
        Returns:
            Response object
        
        Raises:
            APIError: If request fails
        """
        # Add common parameters
        request_params = params.copy()
        if self.api_key:
            request_params["api_key"] = self.api_key
        if self.email:
            request_params["email"] = self.email
        
        return self._make_request(endpoint, request_params, use_cache=use_cache)
    
    def build_search_query(
        self,
        keywords: list[str],
        mesh_terms: Optional[list[str]] = None,
        open_access_only: bool = True,
        has_data_availability: bool = True
    ) -> str:
        """
        Build PubMed search query with keywords and filters.
        
        Args:
            keywords: Search keywords
            mesh_terms: MeSH terms to include
            open_access_only: Filter for open access articles
            has_data_availability: Filter for articles with data availability statements
        
        Returns:
            Formatted search query string
        """
        query_parts = []
        
        # Add keywords
        if keywords:
            keyword_query = " OR ".join(f'"{kw}"[Title/Abstract]' for kw in keywords)
            query_parts.append(f"({keyword_query})")
        
        # Add MeSH terms
        if mesh_terms:
            mesh_query = " OR ".join(f'"{term}"[MeSH Terms]' for term in mesh_terms)
            query_parts.append(f"({mesh_query})")
        
        # Combine with AND
        query = " AND ".join(query_parts) if query_parts else ""
        
        # Add filters
        filters = []
        
        if open_access_only:
            filters.append("open access[filter]")
        
        if has_data_availability:
            filters.append("has data availability statement[filter]")
        
        if filters:
            filter_query = " AND ".join(filters)
            query = f"({query}) AND ({filter_query})" if query else filter_query
        
        return query
    
    def search(
        self,
        keywords: list[str],
        mesh_terms: Optional[list[str]] = None,
        max_results: int = 100,
        open_access_only: bool = True,
        has_data_availability: bool = True
    ) -> list[str]:
        """
        Search PubMed for articles matching criteria.
        
        Args:
            keywords: Search keywords
            mesh_terms: MeSH terms to include
            max_results: Maximum number of results to return
            open_access_only: Filter for open access articles
            has_data_availability: Filter for articles with data availability
        
        Returns:
            List of PubMed IDs (PMIDs)
        """
        query = self.build_search_query(
            keywords=keywords,
            mesh_terms=mesh_terms,
            open_access_only=open_access_only,
            has_data_availability=has_data_availability
        )
        
        logger.info(f"Searching PubMed with query: {query}")
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "xml",
            "usehistory": "y"
        }
        
        try:
            response = self._make_pubmed_request("esearch.fcgi", params)
            return self._parse_search_results(response.text)
        
        except APIError as e:
            logger.error(f"PubMed search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in PubMed search: {e}")
            return []
    
    def _parse_search_results(self, xml_text: str) -> list[str]:
        """
        Parse PubMed search results XML.
        
        Args:
            xml_text: XML response from esearch
        
        Returns:
            List of PMIDs
        """
        try:
            root = ET.fromstring(xml_text)
            id_list = root.find("IdList")
            
            if id_list is None:
                return []
            
            pmids = [id_elem.text for id_elem in id_list.findall("Id") if id_elem.text]
            logger.info(f"Found {len(pmids)} PubMed articles")
            return pmids
        
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed search results: {e}")
            return []
    
    def fetch_article_details(self, pmids: list[str]) -> list[dict[str, Any]]:
        """
        Fetch detailed article information for PMIDs.
        
        Args:
            pmids: List of PubMed IDs
        
        Returns:
            List of article detail dictionaries (empty list on error)
        """
        if not pmids:
            return []
        
        # Fetch in batches of 200 (API limit)
        batch_size = 200
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            pmid_str = ",".join(batch)
            
            params = {
                "db": "pubmed",
                "id": pmid_str,
                "retmode": "xml"
            }
            
            try:
                response = self._make_pubmed_request("efetch.fcgi", params, use_cache=True)
                articles = self._parse_article_details(response.text)
                all_articles.extend(articles)
            
            except APIError as e:
                logger.error(f"Failed to fetch details for batch: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error fetching batch: {e}")
                continue
        
        logger.info(f"Fetched details for {len(all_articles)} articles")
        return all_articles
    
    def _parse_article_details(self, xml_text: str) -> list[dict[str, Any]]:
        """
        Parse article details from PubMed XML.
        
        Args:
            xml_text: XML response from efetch
        
        Returns:
            List of article detail dictionaries
        """
        articles = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for article_elem in root.findall(".//PubmedArticle"):
                try:
                    article = self._extract_article_data(article_elem)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"Failed to parse article: {e}")
                    continue
        
        except ET.ParseError as e:
            logger.error(f"Failed to parse article details XML: {e}")
        
        return articles
    
    def _extract_article_data(self, article_elem: ET.Element) -> Optional[dict[str, Any]]:
        """
        Extract article data from XML element.
        
        Args:
            article_elem: PubmedArticle XML element
        
        Returns:
            Dictionary with article data or None
        """
        medline = article_elem.find(".//MedlineCitation")
        if medline is None:
            return None
        
        # Extract PMID
        pmid_elem = medline.find("PMID")
        pmid = pmid_elem.text if pmid_elem is not None else None
        
        if not pmid:
            return None
        
        # Extract article metadata
        article_node = medline.find("Article")
        if article_node is None:
            return None
        
        # Title
        title_elem = article_node.find(".//ArticleTitle")
        title = title_elem.text if title_elem is not None else ""
        
        # Abstract
        abstract_parts = []
        for abstract_text in article_node.findall(".//AbstractText"):
            if abstract_text.text:
                abstract_parts.append(abstract_text.text)
        abstract = " ".join(abstract_parts)
        
        # Authors
        authors = []
        author_list = article_node.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.find("LastName")
                fore_name = author.find("ForeName")
                if last_name is not None and fore_name is not None:
                    authors.append(f"{fore_name.text} {last_name.text}")
        
        # Publication date
        pub_date_elem = article_node.find(".//PubDate")
        pub_date = self._parse_pub_date(pub_date_elem)
        
        # DOI
        doi = None
        for article_id in article_node.findall(".//ArticleId"):
            if article_id.get("IdType") == "doi":
                doi = article_id.text
                break
        
        # Keywords
        keywords = []
        keyword_list = medline.find("KeywordList")
        if keyword_list is not None:
            for keyword in keyword_list.findall(".//Keyword"):
                if keyword.text:
                    keywords.append(keyword.text)
        
        # MeSH terms
        mesh_list = medline.find("MeshHeadingList")
        if mesh_list is not None:
            for mesh_heading in mesh_list.findall("MeshHeading"):
                descriptor = mesh_heading.find("DescriptorName")
                if descriptor is not None and descriptor.text:
                    keywords.append(descriptor.text)
        
        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "publication_date": pub_date,
            "doi": doi,
            "keywords": keywords,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        }
    
    def _parse_pub_date(self, pub_date_elem: Optional[ET.Element]) -> datetime:
        """
        Parse publication date from XML element.
        
        Args:
            pub_date_elem: PubDate XML element
        
        Returns:
            Publication date or current date if parsing fails
        """
        if pub_date_elem is None:
            return datetime.now()
        
        year_elem = pub_date_elem.find("Year")
        month_elem = pub_date_elem.find("Month")
        day_elem = pub_date_elem.find("Day")
        
        try:
            year = int(year_elem.text) if year_elem is not None else datetime.now().year
            
            # Parse month (can be numeric or text)
            month = 1
            if month_elem is not None and month_elem.text:
                try:
                    month = int(month_elem.text)
                except ValueError:
                    # Month name
                    month_names = {
                        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
                        "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
                        "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
                    }
                    month = month_names.get(month_elem.text[:3], 1)
            
            day = int(day_elem.text) if day_elem is not None else 1
            
            return datetime(year, month, day)
        
        except (ValueError, TypeError):
            return datetime.now()
    
    def convert_to_dataset_source(self, article: dict) -> DatasetSource:
        """
        Convert article dictionary to DatasetSource object.
        
        Args:
            article: Article data dictionary
        
        Returns:
            DatasetSource object
        """
        # Generate source ID from PMID
        source_id = f"pubmed_{article['pmid']}"
        
        return DatasetSource(
            source_id=source_id,
            title=article["title"],
            authors=article["authors"],
            publication_date=article["publication_date"],
            source_type="journal",
            url=article["url"],
            doi=article.get("doi"),
            abstract=article["abstract"],
            keywords=article["keywords"],
            open_access=True,  # Filtered by search
            data_availability="unknown",  # Needs manual verification
            discovery_date=datetime.now(),
            discovery_method="pubmed_search"
        )
    
    def search_and_fetch(
        self,
        keywords: list[str],
        mesh_terms: Optional[list[str]] = None,
        max_results: int = 100
    ) -> list[DatasetSource]:
        """
        Search PubMed and fetch full article details.
        
        Args:
            keywords: Search keywords
            mesh_terms: MeSH terms to include
            max_results: Maximum number of results
        
        Returns:
            List of DatasetSource objects
        """
        # Search for PMIDs
        pmids = self.search(
            keywords=keywords,
            mesh_terms=mesh_terms,
            max_results=max_results
        )
        
        if not pmids:
            logger.info("No PubMed results found")
            return []
        
        # Fetch article details
        articles = self.fetch_article_details(pmids)
        
        # Convert to DatasetSource objects
        sources = [self.convert_to_dataset_source(article) for article in articles]
        
        logger.info(f"Converted {len(sources)} articles to DatasetSource objects")
        return sources
