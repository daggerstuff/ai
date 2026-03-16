"""
Repository API clients for dataset discovery.

Provides clients for:
- Dryad Digital Repository
- Zenodo
- ClinicalTrials.gov
"""

import hashlib
import logging
from datetime import datetime
from typing import Any, Optional

from ..models import DatasetSource
from .base_client import BaseAPIClient, APIError

logger = logging.getLogger(__name__)


class DryadClient(BaseAPIClient):
    """Client for Dryad Digital Repository API."""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize Dryad client.
        
        Args:
            enable_cache: Whether to enable request caching
        """
        super().__init__("dryad", enable_cache=enable_cache)
        
        # Add API key if available
        api_key = self.config.get("api_endpoints.dryad.api_key", "")
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def search_datasets(
        self,
        query: str,
        max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search Dryad for datasets.
        
        Args:
            query: Search query string
            max_results: Maximum number of results
        
        Returns:
            List of dataset dictionaries (empty list on error)
        """
        logger.info(f"Searching Dryad with query: {query}")
        
        params = {
            "q": query,
            "per_page": str(min(max_results, 100))
        }
        
        try:
            response = self._make_request("search", params)
            data = response.json()
            
            results = data.get("_embedded", {}).get("stash:datasets", [])
            logger.info(f"Found {len(results)} Dryad datasets")
            return results
        
        except APIError as e:
            logger.error(f"Dryad search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Dryad search: {e}")
            return []
    
    def get_dataset_details(self, dataset_id: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information for a specific dataset.
        
        Args:
            dataset_id: Dryad dataset ID
        
        Returns:
            Dataset details dictionary or None on error
        """
        try:
            response = self._make_request(f"datasets/{dataset_id}")
            return response.json()
        
        except APIError as e:
            logger.error(f"Failed to fetch Dryad dataset {dataset_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Dryad dataset {dataset_id}: {e}")
            return None
    
    def convert_to_dataset_source(self, dataset: dict) -> Optional[DatasetSource]:
        """
        Convert Dryad dataset to DatasetSource object.
        
        Args:
            dataset: Dryad dataset dictionary
        
        Returns:
            DatasetSource object or None
        """
        try:
            # Extract metadata
            title = dataset.get("title", "")
            abstract = dataset.get("abstract", "")
            
            # Extract authors
            authors = []
            for author in dataset.get("authors", []):
                first_name = author.get("firstName", "")
                last_name = author.get("lastName", "")
                if first_name and last_name:
                    authors.append(f"{first_name} {last_name}")
            
            # Extract dates
            pub_date_str = dataset.get("publicationDate", "")
            try:
                pub_date = datetime.fromisoformat(pub_date_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pub_date = datetime.now()
            
            # Extract identifiers
            doi = dataset.get("identifier", "")
            dataset_id = dataset.get("id", "")
            url = f"https://datadryad.org/stash/dataset/{doi}" if doi else ""
            
            # Extract keywords
            keywords = dataset.get("keywords", [])
            if isinstance(keywords, str):
                keywords = [k.strip() for k in keywords.split(",")]
            
            # Generate source ID
            source_id = f"dryad_{hashlib.md5(dataset_id.encode()).hexdigest()[:12]}"
            
            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type="repository",
                url=url,
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                open_access=True,  # Dryad is open access
                data_availability="available",
                discovery_date=datetime.now(),
                discovery_method="repository_api"
            )
        
        except Exception as e:
            logger.warning(f"Failed to convert Dryad dataset: {e}")
            return None
    
    def search_therapeutic_datasets(
        self,
        keywords: list[str],
        max_results: int = 100
    ) -> list[DatasetSource]:
        """
        Search for therapeutic datasets and convert to DatasetSource objects.
        
        Args:
            keywords: Search keywords
            max_results: Maximum number of results
        
        Returns:
            List of DatasetSource objects
        """
        # Combine keywords into query
        query = " OR ".join(keywords)
        
        # Search datasets
        datasets = self.search_datasets(query, max_results)
        
        # Convert to DatasetSource objects
        sources = []
        for dataset in datasets:
            source = self.convert_to_dataset_source(dataset)
            if source:
                sources.append(source)
        
        logger.info(f"Converted {len(sources)} Dryad datasets to DatasetSource objects")
        return sources


class ZenodoClient(BaseAPIClient):
    """Client for Zenodo API."""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize Zenodo client.
        
        Args:
            enable_cache: Whether to enable request caching
        """
        super().__init__("zenodo", enable_cache=enable_cache)
        
        # Add access token if available
        access_token = self.config.get("api_endpoints.zenodo.access_token", "")
        if access_token:
            self._session.params = {"access_token": access_token}
    
    def search_records(
        self,
        query: str,
        record_type: str = "dataset",
        max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search Zenodo for records.
        
        Args:
            query: Search query string
            record_type: Type of record (dataset, publication, etc.)
            max_results: Maximum number of results
        
        Returns:
            List of record dictionaries (empty list on error)
        """
        logger.info(f"Searching Zenodo with query: {query}")
        
        params = {
            "q": query,
            "type": record_type,
            "size": str(min(max_results, 100))
        }
        
        try:
            response = self._make_request("records", params)
            data = response.json()
            
            results = data.get("hits", {}).get("hits", [])
            logger.info(f"Found {len(results)} Zenodo records")
            return results
        
        except APIError as e:
            logger.error(f"Zenodo search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in Zenodo search: {e}")
            return []
    
    def get_record_details(self, record_id: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information for a specific record.
        
        Args:
            record_id: Zenodo record ID
        
        Returns:
            Record details dictionary or None on error
        """
        try:
            response = self._make_request(f"records/{record_id}")
            return response.json()
        
        except APIError as e:
            logger.error(f"Failed to fetch Zenodo record {record_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching Zenodo record {record_id}: {e}")
            return None
    
    def convert_to_dataset_source(self, record: dict) -> Optional[DatasetSource]:
        """
        Convert Zenodo record to DatasetSource object.
        
        Args:
            record: Zenodo record dictionary
        
        Returns:
            DatasetSource object or None
        """
        try:
            metadata = record.get("metadata", {})
            
            # Extract basic info
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            
            # Extract authors
            authors = []
            for creator in metadata.get("creators", []):
                name = creator.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract publication date
            pub_date_str = metadata.get("publication_date", "")
            try:
                pub_date = datetime.fromisoformat(pub_date_str)
            except (ValueError, AttributeError):
                pub_date = datetime.now()
            
            # Extract identifiers
            doi = metadata.get("doi", "")
            record_id = record.get("id", "")
            url = record.get("links", {}).get("html", f"https://zenodo.org/record/{record_id}")
            
            # Extract keywords
            keywords = metadata.get("keywords", [])
            
            # Check access rights
            access_right = metadata.get("access_right", "")
            open_access = access_right in ["open", "embargoed"]
            
            # Generate source ID
            source_id = f"zenodo_{record_id}"
            
            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type="repository",
                url=url,
                doi=doi,
                abstract=description,
                keywords=keywords,
                open_access=open_access,
                data_availability="available" if open_access else "restricted",
                discovery_date=datetime.now(),
                discovery_method="repository_api"
            )
        
        except Exception as e:
            logger.warning(f"Failed to convert Zenodo record: {e}")
            return None
    
    def search_therapeutic_datasets(
        self,
        keywords: list[str],
        max_results: int = 100
    ) -> list[DatasetSource]:
        """
        Search for therapeutic datasets and convert to DatasetSource objects.
        
        Args:
            keywords: Search keywords
            max_results: Maximum number of results
        
        Returns:
            List of DatasetSource objects
        """
        # Combine keywords into query
        query = " OR ".join(keywords)
        
        # Search records
        records = self.search_records(query, "dataset", max_results)
        
        # Convert to DatasetSource objects
        sources = []
        for record in records:
            source = self.convert_to_dataset_source(record)
            if source:
                sources.append(source)
        
        logger.info(f"Converted {len(sources)} Zenodo records to DatasetSource objects")
        return sources


class ClinicalTrialsClient(BaseAPIClient):
    """Client for ClinicalTrials.gov API."""
    
    def __init__(self, enable_cache: bool = True):
        """
        Initialize ClinicalTrials.gov client.
        
        Args:
            enable_cache: Whether to enable request caching
        """
        super().__init__("clinicaltrials", enable_cache=enable_cache)
    
    def search_studies(
        self,
        query: str,
        status: str = "COMPLETED",
        max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search ClinicalTrials.gov for studies.
        
        Args:
            query: Search query string
            status: Study status filter (COMPLETED, RECRUITING, etc.)
            max_results: Maximum number of results
        
        Returns:
            List of study dictionaries (empty list on error)
        """
        logger.info(f"Searching ClinicalTrials.gov with query: {query}")
        
        params = {
            "query.term": query,
            "filter.overallStatus": status,
            "pageSize": str(min(max_results, 100)),
            "format": "json"
        }
        
        try:
            response = self._make_request("studies", params)
            data = response.json()
            
            results = data.get("studies", [])
            logger.info(f"Found {len(results)} clinical trials")
            return results
        
        except APIError as e:
            logger.error(f"ClinicalTrials.gov search failed: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in ClinicalTrials.gov search: {e}")
            return []
    
    def get_study_details(self, nct_id: str) -> Optional[dict[str, Any]]:
        """
        Get detailed information for a specific study.
        
        Args:
            nct_id: NCT ID (e.g., NCT12345678)
        
        Returns:
            Study details dictionary or None on error
        """
        try:
            response = self._make_request(f"studies/{nct_id}")
            data = response.json()
            return data.get("protocolSection", {})
        
        except APIError as e:
            logger.error(f"Failed to fetch clinical trial {nct_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching clinical trial {nct_id}: {e}")
            return None
    
    def convert_to_dataset_source(self, study: dict) -> Optional[DatasetSource]:
        """
        Convert clinical trial to DatasetSource object.
        
        Args:
            study: Clinical trial dictionary
        
        Returns:
            DatasetSource object or None
        """
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            description = protocol.get("descriptionModule", {})
            contacts = protocol.get("contactsLocationsModule", {})
            
            # Extract basic info
            nct_id = identification.get("nctId", "")
            title = identification.get("officialTitle", "") or identification.get("briefTitle", "")
            brief_summary = description.get("briefSummary", "")
            detailed_description = description.get("detailedDescription", "")
            abstract = f"{brief_summary}\n\n{detailed_description}".strip()
            
            # Extract investigators as authors
            authors = []
            for investigator in contacts.get("centralContacts", []):
                name = investigator.get("name", "")
                if name:
                    authors.append(name)
            
            # Extract dates
            status_module = protocol.get("statusModule", {})
            start_date_struct = status_module.get("startDateStruct", {})
            start_date_str = start_date_struct.get("date", "")
            
            try:
                # Parse date in format "YYYY-MM-DD" or "YYYY-MM" or "YYYY"
                parts = start_date_str.split("-")
                year = int(parts[0]) if len(parts) > 0 else datetime.now().year
                month = int(parts[1]) if len(parts) > 1 else 1
                day = int(parts[2]) if len(parts) > 2 else 1
                pub_date = datetime(year, month, day)
            except (ValueError, IndexError):
                pub_date = datetime.now()
            
            # Extract keywords/conditions
            conditions_module = protocol.get("conditionsModule", {})
            keywords = conditions_module.get("conditions", [])
            keywords.extend(conditions_module.get("keywords", []))
            
            # URL
            url = f"https://clinicaltrials.gov/study/{nct_id}"
            
            # Check for data sharing
            ipd_module = protocol.get("ipdSharingStatementModule", {})
            ipd_sharing = ipd_module.get("ipdSharing", "NO")
            data_available = ipd_sharing in ["YES", "YES_WITH_RESTRICTIONS"]
            
            # Generate source ID
            source_id = f"clinicaltrials_{nct_id}"
            
            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type="clinical_trial",
                url=url,
                doi=None,
                abstract=abstract,
                keywords=keywords,
                open_access=data_available,
                data_availability="available" if data_available else "upon_request",
                discovery_date=datetime.now(),
                discovery_method="repository_api"
            )
        
        except Exception as e:
            logger.warning(f"Failed to convert clinical trial: {e}")
            return None
    
    def search_mental_health_studies(
        self,
        keywords: list[str],
        max_results: int = 100
    ) -> list[DatasetSource]:
        """
        Search for mental health studies and convert to DatasetSource objects.
        
        Args:
            keywords: Search keywords
            max_results: Maximum number of results
        
        Returns:
            List of DatasetSource objects
        """
        # Combine keywords into query
        query = " OR ".join(keywords)
        
        # Search studies
        studies = self.search_studies(query, "COMPLETED", max_results)
        
        # Convert to DatasetSource objects
        sources = []
        for study in studies:
            source = self.convert_to_dataset_source(study)
            if source:
                sources.append(source)
        
        logger.info(f"Converted {len(sources)} clinical trials to DatasetSource objects")
        return sources
