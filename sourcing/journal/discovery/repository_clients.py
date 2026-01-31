"""
Repository API Clients

Implements clients for searching Dryad, Zenodo, and ClinicalTrials.gov
for therapeutic datasets.
"""

import logging
import time
from datetime import datetime
from typing import List, Optional

import requests

from ai.sourcing.journal.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class DryadClient:
    """Client for searching Dryad data repository."""

    def __init__(
        self,
        base_url: str = "https://datadryad.org/api/v2",
        rate_limit_delay: float = 1.0,
        page_size: int = 100,
    ):
        """Initialize Dryad client."""
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.page_size = page_size
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def search(
        self,
        keywords: List[str],
        max_results: int = 100,
    ) -> List[DatasetSource]:
        """
        Search Dryad for datasets.

        Args:
            keywords: Search keywords
            max_results: Maximum number of results

        Returns:
            List of DatasetSource objects
        """
        all_sources = []
        page = 1

        # Build search query
        query = " OR ".join([f'"{kw}"' for kw in keywords])
        # Add therapeutic terms
        therapeutic_terms = ["therapy", "counseling", "psychotherapy", "mental health"]
        query += " OR " + " OR ".join([f'"{term}"' for term in therapeutic_terms])

        try:
            while len(all_sources) < max_results:
                self._rate_limit()

                params = {
                    "q": query,
                    "page": page,
                    "per_page": min(self.page_size, max_results - len(all_sources)),
                }

                url = f"{self.base_url}/datasets"
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                if not isinstance(data, list):
                    if "results" in data:
                        datasets = data["results"]
                    elif "data" in data:
                        datasets = data["data"]
                    else:
                        datasets = []
                else:
                    datasets = data

                if not datasets:
                    break

                for dataset in datasets:
                    try:
                        source = self._parse_dataset(dataset)
                        if source:
                            all_sources.append(source)
                            if len(all_sources) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Error parsing Dryad dataset: {e}")
                        continue

                page += 1
                if len(all_sources) >= max_results:
                    break

            logger.info(f"Found {len(all_sources)} datasets from Dryad")
            return all_sources[:max_results]

        except Exception as e:
            logger.error(f"Error searching Dryad: {e}", exc_info=True)
            return []

    def _parse_dataset(self, dataset: dict) -> Optional[DatasetSource]:
        """Parse a Dryad dataset into a DatasetSource."""
        try:
            # Extract basic info
            title = dataset.get("title", "Untitled")
            doi = dataset.get("doi") or dataset.get("identifier", {}).get("doi")
            url = dataset.get("url") or dataset.get("identifier", {}).get("url")
            if not url and doi:
                url = f"https://doi.org/{doi}"

            # Extract authors
            authors = []
            author_list = dataset.get("authors", [])
            for author in author_list:
                if isinstance(author, dict):
                    name = author.get("name") or author.get("fullName", "")
                    if name:
                        authors.append(name)
                elif isinstance(author, str):
                    authors.append(author)

            # Extract publication date
            pub_date = self._parse_date(dataset)

            # Extract abstract/description
            abstract = dataset.get("abstract") or dataset.get("description", "")

            # Extract keywords
            keywords = []
            keyword_list = dataset.get("keywords", [])
            for keyword in keyword_list:
                if isinstance(keyword, str):
                    keywords.append(keyword)
                elif isinstance(keyword, dict):
                    kw = keyword.get("name") or keyword.get("value", "")
                    if kw:
                        keywords.append(kw)

            # Generate source ID
            source_id = f"dryad_{dataset.get('id', hash(title))}"

            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type="repository",
                url=url or "https://datadryad.org/",
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                open_access=True,  # Dryad is open access
                data_availability="available",  # Dryad datasets are available
                discovery_date=datetime.now(),
                discovery_method="repository_api",
            )

        except Exception as e:
            logger.warning(f"Error parsing Dryad dataset: {e}")
            return None

    def _parse_date(self, dataset: dict) -> datetime:
        """Parse publication date."""
        try:
            date_str = dataset.get("publicationDate") or dataset.get("datePublished")
            if date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            pass

        return datetime.now()


class ZenodoClient:
    """Client for searching Zenodo research repository."""

    def __init__(
        self,
        base_url: str = "https://zenodo.org/api",
        rate_limit_delay: float = 1.0,
        page_size: int = 100,
    ):
        """Initialize Zenodo client."""
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.page_size = page_size
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def search(
        self,
        keywords: List[str],
        max_results: int = 100,
    ) -> List[DatasetSource]:
        """
        Search Zenodo for datasets.

        Args:
            keywords: Search keywords
            max_results: Maximum number of results

        Returns:
            List of DatasetSource objects
        """
        all_sources = []
        page = 1

        # Build search query
        query = " OR ".join([f'"{kw}"' for kw in keywords])
        # Add therapeutic terms
        therapeutic_terms = ["therapy", "counseling", "psychotherapy", "mental health"]
        query += " OR " + " OR ".join([f'"{term}"' for term in therapeutic_terms])
        # Filter for datasets
        query += " AND resource_type:dataset"

        try:
            while len(all_sources) < max_results:
                self._rate_limit()

                params = {
                    "q": query,
                    "page": page,
                    "size": min(self.page_size, max_results - len(all_sources)),
                }

                url = f"{self.base_url}/records"
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                hits = data.get("hits", {}).get("hits", [])
                if not hits:
                    break

                for hit in hits:
                    try:
                        source = self._parse_record(hit)
                        if source:
                            all_sources.append(source)
                            if len(all_sources) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Error parsing Zenodo record: {e}")
                        continue

                # Check if there are more pages
                total = data.get("hits", {}).get("total", 0)
                if len(all_sources) >= total or len(all_sources) >= max_results:
                    break

                page += 1

            logger.info(f"Found {len(all_sources)} datasets from Zenodo")
            return all_sources[:max_results]

        except Exception as e:
            logger.error(f"Error searching Zenodo: {e}", exc_info=True)
            return []

    def _parse_record(self, record: dict) -> Optional[DatasetSource]:
        """Parse a Zenodo record into a DatasetSource."""
        try:
            metadata = record.get("metadata", {})
            id = record.get("id", "")

            # Extract basic info
            title = metadata.get("title", "Untitled")
            if isinstance(title, list):
                title = title[0] if title else "Untitled"

            # Extract authors
            authors = []
            creator_list = metadata.get("creators", [])
            for creator in creator_list:
                if isinstance(creator, dict):
                    name = creator.get("name", "")
                    if name:
                        authors.append(name)
                elif isinstance(creator, str):
                    authors.append(creator)

            # Extract publication date
            pub_date = self._parse_date(metadata)

            # Extract DOI
            doi = None
            doi_list = metadata.get("doi", "")
            if doi_list:
                doi = doi_list if isinstance(doi_list, str) else doi_list[0] if isinstance(doi_list, list) else None

            # Extract abstract/description
            abstract = metadata.get("description", "")
            if isinstance(abstract, list):
                abstract = abstract[0] if abstract else ""

            # Extract keywords
            keywords = []
            keyword_list = metadata.get("keywords", [])
            for keyword in keyword_list:
                if isinstance(keyword, str):
                    keywords.append(keyword)

            # Extract URL
            links = record.get("links", {})
            url = links.get("html") or links.get("self", f"https://zenodo.org/record/{id}")

            # Generate source ID
            source_id = f"zenodo_{id}"

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
                open_access=True,  # Zenodo is open access
                data_availability="available",  # Zenodo datasets are available
                discovery_date=datetime.now(),
                discovery_method="repository_api",
            )

        except Exception as e:
            logger.warning(f"Error parsing Zenodo record: {e}")
            return None

    def _parse_date(self, metadata: dict) -> datetime:
        """Parse publication date."""
        try:
            date_str = metadata.get("publication_date") or metadata.get("date")
            if date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        except (ValueError, TypeError, AttributeError):
            pass

        return datetime.now()


class ClinicalTrialsClient:
    """Client for searching ClinicalTrials.gov."""

    def __init__(
        self,
        base_url: str = "https://clinicaltrials.gov/api/v2",
        rate_limit_delay: float = 1.0,
        page_size: int = 100,
    ):
        """Initialize ClinicalTrials.gov client."""
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.page_size = page_size
        self.last_request_time = 0.0

    def _rate_limit(self) -> None:
        """Enforce rate limiting."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()

    def search(
        self,
        keywords: List[str],
        condition: Optional[str] = None,
        status: str = "COMPLETED",
        max_results: int = 100,
    ) -> List[DatasetSource]:
        """
        Search ClinicalTrials.gov for completed studies.

        Args:
            keywords: Search keywords
            condition: Optional medical condition filter
            status: Study status (default: COMPLETED)
            max_results: Maximum number of results

        Returns:
            List of DatasetSource objects
        """
        all_sources = []
        page = 1

        # Build search query
        query_parts = []
        for keyword in keywords:
            query_parts.append(f'"{keyword}"')

        if condition:
            query_parts.append(f'CONDITION:"{condition}"')

        query_parts.append(f'STATUS:{status}')

        # Add therapeutic terms
        therapeutic_terms = ["therapy", "counseling", "psychotherapy", "mental health"]
        for term in therapeutic_terms:
            query_parts.append(f'"{term}"')

        query = " AND ".join(query_parts)

        try:
            while len(all_sources) < max_results:
                self._rate_limit()

                params = {
                    "query.cond": condition or "",
                    "query.term": " OR ".join(keywords),
                    "filter.overallStatus": status,
                    "pageSize": min(self.page_size, max_results - len(all_sources)),
                    "page": page,
                }

                url = f"{self.base_url}/studies"
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()

                studies = data.get("studies", [])
                if not studies:
                    break

                for study in studies:
                    try:
                        source = self._parse_study(study)
                        if source:
                            all_sources.append(source)
                            if len(all_sources) >= max_results:
                                break
                    except Exception as e:
                        logger.warning(f"Error parsing ClinicalTrials study: {e}")
                        continue

                # Check if there are more pages
                total = data.get("totalCount", 0)
                if len(all_sources) >= total or len(all_sources) >= max_results:
                    break

                page += 1

            logger.info(f"Found {len(all_sources)} studies from ClinicalTrials.gov")
            return all_sources[:max_results]

        except Exception as e:
            logger.error(f"Error searching ClinicalTrials.gov: {e}", exc_info=True)
            return []

    def _parse_study(self, study: dict) -> Optional[DatasetSource]:
        """Parse a ClinicalTrials.gov study into a DatasetSource."""
        try:
            protocol = study.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})

            # Extract basic info
            title = identification.get("briefTitle", "Untitled")
            nct_id = identification.get("nctId", "")

            # Extract authors/investigators
            authors = []
            contacts = protocol.get("contactsLocationsModule", {})
            contact_list = contacts.get("centralContacts", [])
            for contact in contact_list:
                name = contact.get("name", "")
                if name:
                    authors.append(name)

            # Extract publication date
            pub_date = self._parse_date(status)

            # Extract description
            description = protocol.get("descriptionModule", {})
            abstract = description.get("briefSummary", "")

            # Extract keywords
            keywords = []
            conditions = protocol.get("conditionsModule", {}).get("conditions", [])
            keywords.extend(conditions)

            # Extract URL
            url = f"https://clinicaltrials.gov/study/{nct_id}"

            # Check data availability
            data_availability = "upon_request"  # Clinical trials typically require request

            # Generate source ID
            source_id = f"clinical_trial_{nct_id}"

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
                open_access=True,  # ClinicalTrials.gov is open access
                data_availability=data_availability,
                discovery_date=datetime.now(),
                discovery_method="repository_api",
            )

        except Exception as e:
            logger.warning(f"Error parsing ClinicalTrials study: {e}")
            return None

    def _parse_date(self, status: dict) -> datetime:
        """Parse study completion date."""
        try:
            completion_date = status.get("completionDateStruct", {})
            if completion_date:
                year = completion_date.get("year")
                month = completion_date.get("month", 1)
                day = completion_date.get("day", 1)
                if year:
                    return datetime(int(year), int(month), int(day))
        except (ValueError, TypeError, AttributeError):
            pass

        return datetime.now()

